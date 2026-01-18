from functools import partial
from typing import Any, Callable, Dict, List, Type, TypeVar, Union, overload
import torch
import torch.nn as nn
import torch.types
def dict_multimap(fn: Callable[[list], Any], dicts: List[dict]) -> dict:
    first = dicts[0]
    new_dict = {}
    for k, v in first.items():
        all_v = [d[k] for d in dicts]
        if isinstance(v, dict):
            new_dict[k] = dict_multimap(fn, all_v)
        else:
            new_dict[k] = fn(all_v)
    return new_dict