import collections
import functools
import torch
from typing import Any
@functools.wraps(bwd)
def decorate_bwd(*args, **kwargs):
    with autocast(enabled=args[0]._fwd_used_autocast, dtype=args[0]._dtype):
        return bwd(*args, **kwargs)