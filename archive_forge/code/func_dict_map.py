from functools import partial
from typing import Any, Callable, Dict, List, Type, TypeVar, Union, overload
import torch
import torch.nn as nn
import torch.types
def dict_map(fn: Callable[[T], Any], dic: Dict[Any, Union[dict, list, tuple, T]], leaf_type: Type[T]) -> Dict[Any, Union[dict, list, tuple, Any]]:
    new_dict: Dict[Any, Union[dict, list, tuple, Any]] = {}
    for k, v in dic.items():
        if isinstance(v, dict):
            new_dict[k] = dict_map(fn, v, leaf_type)
        else:
            new_dict[k] = tree_map(fn, v, leaf_type)
    return new_dict