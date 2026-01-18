from parlai.core.agents import Agent
from parlai.core.dict import DictionaryAgent
from parlai.utils.misc import maintain_dialog_history, load_cands
from parlai.core.torch_agent import TorchAgent
from .modules import Starspace
import torch
from torch import optim
import torch.nn as nn
from collections import deque
import copy
import os
import random
import json
def dict_neighbors(self, word, useRHS=False):
    input = self.t2v(word)
    W = self.model.encoder.lt.weight
    q = W[input.data[0][0]]
    if useRHS:
        W = self.model.encoder2.lt.weight
    score = torch.Tensor(W.size(0))
    for i in range(W.size(0)):
        score[i] = torch.nn.functional.cosine_similarity(q, W[i], dim=0).item()
    val, ind = score.sort(descending=True)
    for i in range(20):
        print(str(ind[i]) + ' [' + str(val[i]) + ']: ' + self.v2t(torch.Tensor([ind[i]])))