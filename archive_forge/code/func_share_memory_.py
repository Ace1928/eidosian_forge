import torch
from torch._C import _disabled_torch_function_impl
from collections import OrderedDict
def share_memory_(self):
    raise RuntimeError("Can't share memory on an uninitialized parameter or buffer. Call `forward` to initialize the parameters before calling `module.share_memory()`.")