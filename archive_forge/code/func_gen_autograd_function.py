import torch
import torch.utils._pytree as pytree
from collections import namedtuple
import functools
def gen_autograd_function(name, forward, backward):
    generated_cls = type(name, (torch.autograd.Function,), {'forward': staticmethod(forward), 'backward': staticmethod(backward)})
    return generated_cls