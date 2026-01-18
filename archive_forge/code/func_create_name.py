import collections
from collections import defaultdict
from .node import Node, Argument, Target, map_arg, _type_repr, _get_qualified_name
import torch.utils._pytree as pytree
from . import _pytree as fx_pytree
from ._compatibility import compatibility
import contextlib
from typing import TYPE_CHECKING, Callable, Any, List, Dict, NamedTuple, Optional, Tuple, Set, FrozenSet, Type
from dataclasses import dataclass
from contextlib import contextmanager
import copy
import enum
import torch
import keyword
import re
import builtins
import math
import warnings
import inspect
def create_name(self, candidate: str, obj: Optional[Any]) -> str:
    """Create a unique name.

        Arguments:
            candidate: used as the basis for the unique name, relevant to the user.
            obj: If not None, an object that will be associated with the unique name.
        """
    if obj is not None and obj in self._obj_to_name:
        return self._obj_to_name[obj]
    candidate = self._illegal_char_regex.sub('_', candidate)
    if not candidate:
        candidate = '_unnamed'
    if candidate[0].isdigit():
        candidate = f'_{candidate}'
    match = self._name_suffix_regex.match(candidate)
    if match is None:
        base = candidate
        num = None
    else:
        base, num_str = match.group(1, 2)
        num = int(num_str)
    candidate = base if num is None else f'{base}_{num}'
    if not num:
        num = self._base_count[base]
    while candidate in self._used_names or self._is_illegal_name(candidate, obj):
        num += 1
        candidate = f'{base}_{num}'
    self._used_names.add(candidate)
    self._base_count[base] = num
    if obj is None:
        self._unassociated_names.add(candidate)
    else:
        self._obj_to_name[obj] = candidate
    return candidate