import torch
from typing import Any, Dict, List, Optional, Sequence, Tuple, TypeVar, Union, overload
from ._functions import Scatter, Gather
import warnings
def gather_map(outputs):
    out = outputs[0]
    if isinstance(out, torch.Tensor):
        return Gather.apply(target_device, dim, *outputs)
    if out is None:
        return None
    if isinstance(out, dict):
        if not all((len(out) == len(d) for d in outputs)):
            raise ValueError('All dicts must have the same number of keys')
        return type(out)(((k, gather_map([d[k] for d in outputs])) for k in out))
    if _is_namedtuple(out):
        return type(out)._make(map(gather_map, zip(*outputs)))
    return type(out)(map(gather_map, zip(*outputs)))