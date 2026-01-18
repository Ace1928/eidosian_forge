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
def feedback_step(self, batch):
    batchsize = batch.text_vec.size(0)
    warn_once('WARNING: feedback candidates are hardcoded to batch')
    if self.model.training:
        cands, cand_vecs, label_inds = self._build_candidates(batch, source='batch', mode='train')
    else:
        cands, cand_vecs, label_inds = self._build_candidates(batch, source='batch', mode='eval')
    scores = self.model.score_feedback(batch.text_vec, cand_vecs)
    _, ranks = scores.sort(1, descending=True)
    if self.model.training:
        cand_ranked = None
        preds = [cands[ordering[0]] for ordering in ranks]
    else:
        cand_ranked = []
        for ordering in ranks:
            cand_ranked.append([cands[rank] for rank in ordering])
        preds = [cand_ranked[i][0] for i in range(batchsize)]
    if label_inds is None:
        loss = None
    else:
        loss = self.criterion(scores, label_inds).mean()
        self.update_fee_metrics(loss, ranks, label_inds, batchsize)
    return (loss, preds, cand_ranked)