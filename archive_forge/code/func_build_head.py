import torch
import torch.nn as nn
from parlai.agents.transformer.modules import TransformerEncoder
def build_head(self, opt, outdim=1, num_layers=1):
    dim = self.opt['embedding_size']
    modules = []
    for _ in range(num_layers - 1):
        modules.append(nn.Linear(dim, dim))
        modules.append(nn.ReLU())
    modules.append(nn.Linear(dim, outdim))
    return nn.Sequential(*modules)