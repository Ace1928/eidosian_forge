import torch
import torch.nn as nn
from parlai.agents.transformer.modules import TransformerEncoder
def score_satisfaction(self, x_vecs):
    return torch.sigmoid(self.x_sat_head(self.x_sat_encoder(x_vecs))).squeeze(1)