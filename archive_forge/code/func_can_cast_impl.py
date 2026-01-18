from collections import namedtuple
import torch
from . import _casting_dicts as _cd
def can_cast_impl(from_torch_dtype, to_torch_dtype, casting):
    return _cd._can_cast_dict[casting][from_torch_dtype][to_torch_dtype]