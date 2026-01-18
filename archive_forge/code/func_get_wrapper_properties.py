import torch
from copy import deepcopy
from torch.utils._pytree import tree_map
from torch.testing._internal.logging_tensor import LoggingTensor
@classmethod
def get_wrapper_properties(cls, size, values, indices, requires_grad=False):
    assert values.device == indices.device
    return (values, {'size': size, 'requires_grad': requires_grad})