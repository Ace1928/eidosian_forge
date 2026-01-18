import torch
from torch._C import _disabled_torch_function_impl
from collections import OrderedDict
def is_lazy(param):
    return isinstance(param, UninitializedTensorMixin)