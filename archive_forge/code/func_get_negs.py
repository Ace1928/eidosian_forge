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
def get_negs(self, xs, ys):
    negs = []
    cache_sz = len(self.ys_cache) - 1
    if cache_sz < 1:
        return negs
    k = self.opt['neg_samples']
    for _i in range(1, k * 3):
        index = random.randint(0, cache_sz)
        neg = self.ys_cache[index]
        if not self.same(ys, neg):
            negs.append(neg)
            if len(negs) >= k:
                break
    if self.opt['parrot_neg'] > 0:
        utt = self.history['last_utterance']
        if len(utt) > 2:
            query = torch.LongTensor(utt).unsqueeze(0)
            negs.append(query)
    return negs