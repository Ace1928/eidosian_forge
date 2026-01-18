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
def predict_satisfaction_by_uncertainty(self, batch):
    assert self.opt['history_size'] > 2
    text_vecs = []
    cand_vecs = []
    for vec in batch.text_vec:
        last_p1 = (vec == self.dict.txt2vec('__p1__')[0]).nonzero()[-1].item()
        last_p2 = (vec == self.dict.txt2vec('__p2__')[0]).nonzero()[-1].item()
        text_vecs.append(vec[:last_p2])
        cand_vecs.append(vec[last_p2 + 1:last_p1])
    text_padded, _ = padded_tensor(text_vecs)
    cand_padded, _ = padded_tensor(cand_vecs)
    scores = self.model.score_dialog(text_padded, cand_padded)
    confidences = F.softmax(scores, dim=1).cpu().detach().numpy()
    preds = []
    for example in confidences:
        ranked_confidences = sorted(list(example), reverse=True)
        if self.opt['uncertainty_style'] == 'mag':
            mag = ranked_confidences[0]
            preds.append(mag > self.opt['uncertainty_threshold'])
        elif self.opt['uncertainty_style'] == 'gap':
            gap = ranked_confidences[0] - ranked_confidences[1]
            preds.append(gap > self.opt['uncertainty_threshold'])
    loss = torch.tensor(0)
    preds = torch.LongTensor(preds)
    labels = torch.LongTensor([int(l) == 1 for l in batch.labels])
    batchsize = len(labels)
    self.update_sat_metrics(loss, preds, labels, batchsize)
    return preds