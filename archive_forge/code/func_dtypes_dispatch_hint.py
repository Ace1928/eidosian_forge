import collections
import warnings
from functools import partial, wraps
from typing import Sequence
import numpy as np
import torch
from torch.testing._internal.common_cuda import TEST_CUDA
from torch.testing._internal.common_dtype import (
from torch.testing._internal.common_utils import torch_to_numpy_dtype_dict
def dtypes_dispatch_hint(dtypes):
    return_type = collections.namedtuple('return_type', 'dispatch_fn dispatch_fn_str')
    if len(dtypes) == 0:
        return return_type((), str(tuple()))
    set_dtypes = set(dtypes)
    for dispatch in COMPLETE_DTYPES_DISPATCH:
        if set(dispatch()) == set_dtypes:
            return return_type(dispatch, dispatch.__name__ + '()')
    chosen_dispatch = None
    chosen_dispatch_score = 0.0
    for dispatch in EXTENSIBLE_DTYPE_DISPATCH:
        dispatch_dtypes = set(dispatch())
        if not dispatch_dtypes.issubset(set_dtypes):
            continue
        score = len(dispatch_dtypes)
        if score > chosen_dispatch_score:
            chosen_dispatch_score = score
            chosen_dispatch = dispatch
    if chosen_dispatch is None:
        return return_type((), str(dtypes))
    return return_type(partial(dispatch, *tuple(set(dtypes) - set(dispatch()))), dispatch.__name__ + str(tuple(set(dtypes) - set(dispatch()))))