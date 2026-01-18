from abc import ABC, abstractmethod
from collections import UserDict
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch
from ..utils import add_start_docstrings
from .beam_constraints import Constraint, ConstraintListState
def step_sentence_constraint(self, batch_idx: int, input_ids: torch.LongTensor, vocab_scores: torch.FloatTensor, sent_beam_scores: torch.FloatTensor, sent_beam_tokens: torch.LongTensor, sent_beam_indices: torch.LongTensor, push_progress: bool=False):
    orig_len = sent_beam_indices.size(0)
    device = sent_beam_indices.device
    topk_contraint_states = self.make_constraint_states(orig_len)
    advance_constraint_states = self.make_constraint_states(orig_len)
    sidx, eidx = (batch_idx * orig_len, (batch_idx + 1) * orig_len)
    this_batch_input_ids = input_ids[sidx:eidx]
    this_batch_token_scores = vocab_scores[sidx:eidx]
    full_hypotheses = torch.cat((input_ids[sent_beam_indices], sent_beam_tokens.unsqueeze(-1)), dim=-1)
    track_new = {'new_seqs': full_hypotheses.tolist(), 'new_states': [], 'new_indices': [], 'new_tokens': [], 'new_scores': []}
    for seq_idx, pre_seq in enumerate(this_batch_input_ids):
        topk_state = topk_contraint_states[seq_idx]
        topk_state.reset(full_hypotheses[seq_idx].cpu().tolist())
        advance_state = advance_constraint_states[seq_idx]
        advance_state.reset(pre_seq.cpu().tolist())
        if not advance_state.completed:
            advance_tokens = torch.LongTensor(advance_state.advance()).to(device)
            for advance_token in advance_tokens:
                new_state = advance_state.copy(stateful=True)
                new_state.add(advance_token.cpu().tolist())
                advance_seq = torch.cat((pre_seq, advance_token.unsqueeze(0)), -1).cpu().tolist()
                if advance_seq not in track_new['new_seqs']:
                    track_new['new_seqs'].append(advance_seq)
                    track_new['new_indices'].append(sidx + seq_idx)
                    track_new['new_tokens'].append(advance_token)
                    track_new['new_scores'].append(this_batch_token_scores[seq_idx].take(advance_token))
                    track_new['new_states'].append(new_state)
        elif push_progress:
            new_score, new_token = torch.max(this_batch_token_scores[seq_idx], 0)
            advance_seq = torch.cat((pre_seq, new_token.unsqueeze(0)), -1)
            advance_state = advance_constraint_states[seq_idx]
            advance_seq = advance_seq.cpu().tolist()
            advance_state.reset(advance_seq)
            if advance_seq not in track_new['new_seqs']:
                track_new['new_seqs'].append(advance_seq)
                track_new['new_indices'].append(seq_idx)
                track_new['new_tokens'].append(new_token)
                track_new['new_scores'].append(new_score)
                track_new['new_states'].append(advance_state)
    if len(track_new['new_indices']) > 0:
        new_indices = torch.tensor(track_new['new_indices']).to(device)
        new_tokens = torch.stack(track_new['new_tokens']).to(device)
        new_scores = torch.stack(track_new['new_scores']).to(device)
        all_states = topk_contraint_states + track_new['new_states']
        all_tokens = torch.cat((sent_beam_tokens, new_tokens), -1)
        all_scores = torch.cat((sent_beam_scores, new_scores), -1)
        all_banks = torch.tensor([one.get_bank() for one in all_states]).to(device)
        zipped = all_banks * 100 + all_scores
        indices = zipped.sort(descending=True).indices
        sorted_banks = all_banks[indices]
        counter = -1
        cur_bank = sorted_banks[0]
        increments = []
        for bank in sorted_banks:
            if bank == cur_bank:
                counter += 1
            else:
                counter = 0
                cur_bank = bank
            increments.append(counter)
        rearrangers = torch.tensor(np.argsort(increments, kind='mergesort'))
        indices = indices[rearrangers][:orig_len]
        sent_beam_scores = all_scores[indices]
        sent_beam_tokens = all_tokens[indices]
        sent_beam_indices = torch.cat((sent_beam_indices, new_indices))[indices]
    return (sent_beam_scores, sent_beam_tokens, sent_beam_indices)