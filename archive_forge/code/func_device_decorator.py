import torch
from torch.overrides import TorchFunctionMode
from torch.utils._contextlib import context_decorator
import functools
def device_decorator(device, func):
    return context_decorator(lambda: device, func)