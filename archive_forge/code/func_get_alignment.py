import math
import os
from typing import List, Optional, Tuple
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import LayerNorm as FusedLayerNorm
from ...activations import ACT2FN
from ...modeling_utils import PreTrainedModel
from ...utils import add_start_docstrings, logging
from ...utils.logging import tqdm
from .configuration_jukebox import ATTENTION_PATTERNS, JukeboxConfig, JukeboxPriorConfig, JukeboxVQVAEConfig
def get_alignment(music_tokens, labels, prior, config):
    level = prior.levels - 1
    n_ctx = prior.n_ctx
    tokens = music_tokens[level]
    batch_size, total_length = (tokens.shape[0], tokens.shape[1])
    if total_length < n_ctx:
        padding_length = n_ctx - total_length
        tokens = torch.cat([tokens, torch.zeros(batch_size, n_ctx - total_length, dtype=tokens.dtype, device=tokens.device)], dim=1)
        total_length = tokens.shape[1]
    else:
        padding_length = 0
    hop_length = int(config.hop_fraction[-level - 1] * prior.n_ctx)
    alignment_head, alignment_layer = (config.prior_alignment_head[0], config.prior_alignment_layer[0])
    attn_layers = {alignment_layer}
    alignment_hops = {}
    indices_hops = {}
    for start in tqdm(get_starts(total_length, n_ctx, hop_length), desc='Computing lyric to music alignment '):
        end = start + n_ctx
        metadata, indices_hop = prior.get_metadata(labels, start, config.sample_length, get_indices=True, offset=0)
        tokens_bs = torch.chunk(tokens, batch_size, dim=0)
        metadata_bs = torch.chunk(metadata, batch_size, dim=0)
        w_hops = []
        for tokens_i, metadata_i in zip(tokens_bs, metadata_bs):
            w_hop = prior.forward_tokens(tokens_i[:, start:end], [], metadata_i, get_attn_weights=attn_layers)
            w_hops.append(w_hop[0][:, alignment_head])
            del w_hop
        weights = torch.cat(w_hops, dim=0)
        del w_hops
        alignment_hop = weights.float().cpu().numpy()
        del weights
        indices_hops[start] = indices_hop
        alignment_hops[start] = alignment_hop
    alignments = []
    for item in range(batch_size):
        full_tokens = labels[0, 3:]
        alignment = np.zeros((total_length, len(full_tokens) + 1))
        for start in reversed(get_starts(total_length, n_ctx, hop_length)):
            end = start + n_ctx
            alignment_hop = alignment_hops[start][item]
            indices = indices_hops[start][item]
            alignment[start:end, indices] = alignment_hop
        alignment = alignment[:total_length - padding_length, :-1]
        alignments.append(alignment)
    return alignments