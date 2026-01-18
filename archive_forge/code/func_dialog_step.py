import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from parlai.core.torch_agent import TorchAgent, Output
from parlai.utils.misc import round_sigfigs, warn_once
from parlai.utils.torch import padded_tensor
from parlai.agents.transformer.transformer import TransformerRankerAgent
from .feedback_classifier.feedback_classifier import FeedbackClassifierRegex
from .modules import SelfFeedingModel
def dialog_step(self, batch):
    batchsize = batch.text_vec.size(0)
    if self.model.training:
        cands, cand_vecs, label_inds = self._build_candidates(batch, source=self.opt['candidates'], mode='train')
    else:
        cands, cand_vecs, label_inds = self._build_candidates(batch, source=self.opt['eval_candidates'], mode='eval')
        if self.encode_candidate_vecs and self.eval_candidates in ['fixed', 'vocab']:
            if self.eval_candidates == 'fixed':
                cand_vecs = self.fixed_candidate_encs
            elif self.eval_candidates == 'vocab':
                cand_vecs = self.vocab_candidate_encs
    scores = self.model.score_dialog(batch.text_vec, cand_vecs)
    _, ranks = scores.sort(1, descending=True)
    if self.model.training:
        cand_ranked = None
        if cand_vecs.dim() == 2:
            preds = [cands[ordering[0]] for ordering in ranks]
        elif cand_vecs.dim() == 3:
            preds = [cands[i][ordering[0]] for i, ordering in enumerate(ranks)]
    else:
        cand_ranked = []
        for i, ordering in enumerate(ranks):
            if cand_vecs.dim() == 2:
                cand_list = cands
            elif cand_vecs.dim() == 3:
                cand_list = cands[i]
            cand_ranked.append([cand_list[rank] for rank in ordering])
        preds = [cand_ranked[i][0] for i in range(batchsize)]
    if label_inds is None:
        loss = None
    else:
        loss = self.criterion(scores, label_inds).mean()
        self.update_dia_metrics(loss, ranks, label_inds, batchsize)
    return (loss, preds, cand_ranked)