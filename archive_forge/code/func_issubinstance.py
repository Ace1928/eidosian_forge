import collections
import functools
import numbers
import sys
from torch.utils.data.datapipes._hook_iterator import hook_iterator, _SnapshotState
from typing import (Any, Dict, Iterator, Generic, List, Set, Tuple, TypeVar, Union,
from typing import _eval_type, _tp_cache, _type_check, _type_repr  # type: ignore[attr-defined]
from typing import ForwardRef
from abc import ABCMeta
from typing import _GenericAlias  # type: ignore[attr-defined, no-redef]
def issubinstance(data, data_type):
    if not issubtype(type(data), data_type, recursive=False):
        return False
    dt_args = getattr(data_type, '__args__', None)
    if isinstance(data, tuple):
        if dt_args is None or len(dt_args) == 0:
            return True
        if len(dt_args) != len(data):
            return False
        return all((issubinstance(d, t) for d, t in zip(data, dt_args)))
    elif isinstance(data, (list, set)):
        if dt_args is None or len(dt_args) == 0:
            return True
        t = dt_args[0]
        return all((issubinstance(d, t) for d in data))
    elif isinstance(data, dict):
        if dt_args is None or len(dt_args) == 0:
            return True
        kt, vt = dt_args
        return all((issubinstance(k, kt) and issubinstance(v, vt) for k, v in data.items()))
    return True