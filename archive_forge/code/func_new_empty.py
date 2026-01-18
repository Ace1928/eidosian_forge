import torch
from copy import deepcopy
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import LoggingTensor
def new_empty(self, shape):
    return type(self)(torch.empty(shape))