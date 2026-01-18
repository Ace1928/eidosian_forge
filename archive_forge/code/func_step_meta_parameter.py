import inspect
import re
import warnings
from typing import Any, Dict
import torch
from torch.testing import make_tensor
def step_meta_parameter(name, value, direction, meta, m=m, n=n, k=k, bm=bm, bk=bk):
    is_log = name in {'SPLIT_N', 'num_warps'}
    min_value = dict(SPLIT_N=1, num_warps=1, num_stages=1, GROUP_SIZE_ROW=1)[name]
    max_value = dict(SPLIT_N=max(n // bm, 1)).get(name)
    value_step = dict(SPLIT_N=2, num_warps=2, num_stages=1, GROUP_SIZE_ROW=1)[name]
    if is_log:
        next_value = value * value_step ** direction if direction > 0 else value // value_step ** abs(direction)
    else:
        next_value = value + value_step * direction
    if min_value is not None:
        next_value = max(next_value, min_value)
    if max_value is not None:
        next_value = min(next_value, max_value)
    if name == 'SPLIT_N' and n % next_value != 0:
        return value
    return next_value