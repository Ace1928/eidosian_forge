from collections import namedtuple
import torch
from . import _casting_dicts as _cd
def is_scalar_or_symbolic(x):
    return isinstance(x, _SCALAR_AND_SYMBOLIC_TYPES)