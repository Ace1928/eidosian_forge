from collections import namedtuple
import torch
from . import _casting_dicts as _cd
def result_type_impl(*tensors):
    dtyp = tensors[0].dtype
    if len(tensors) == 1:
        return dtyp
    for curr in tensors[1:]:
        dtyp = _cd._result_type_dict[dtyp][curr.dtype]
    return dtyp