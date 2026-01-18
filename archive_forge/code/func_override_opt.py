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
def override_opt(self, new_opt):
    """
        Set overridable opts from loaded opt file.

        Print out each added key and each overriden key. Only override args specific to
        the model.
        """
    model_args = {'embeddingsize', 'optimizer'}
    for k, v in new_opt.items():
        if k not in model_args:
            continue
        if k not in self.opt:
            print('Adding new option [ {k}: {v} ]'.format(k=k, v=v))
        elif self.opt[k] != v:
            print('Overriding option [ {k}: {old} => {v}]'.format(k=k, old=self.opt[k], v=v))
        self.opt[k] = v
    return self.opt