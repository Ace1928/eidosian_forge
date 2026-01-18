import torch
from torch.nn.modules.container import ModuleList, ModuleDict, Module
from torch.nn.parameter import Parameter
from torch import Tensor
import collections
import copyreg
from copy import deepcopy
from contextlib import contextmanager
from typing import Union, Optional, Dict, Tuple, Sequence
def default_deepcopy(self, memo):
    obj = memo.get(id(self), None)
    if obj is not None:
        return obj
    replica = self.__new__(self.__class__)
    memo[id(self)] = replica
    replica.__dict__ = deepcopy(self.__dict__, memo)
    slots_to_save = copyreg._slotnames(self.__class__)
    for slot in slots_to_save:
        if hasattr(self, slot):
            setattr(replica, slot, deepcopy(getattr(self, slot), memo))
    return replica