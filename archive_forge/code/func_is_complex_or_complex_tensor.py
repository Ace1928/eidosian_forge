from collections import namedtuple
import torch
from . import _casting_dicts as _cd
def is_complex_or_complex_tensor(x):
    return _dtype_for_scalar_or_tensor(x).is_complex