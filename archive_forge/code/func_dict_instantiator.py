import collections.abc
import dataclasses
import enum
import inspect
import os
import pathlib
from collections import deque
from typing import (
from typing_extensions import Annotated, Final, Literal, get_args, get_origin
from . import _resolver
from . import _strings
from ._typing import TypeForm
from .conf import _markers
def dict_instantiator(strings: List[str]) -> Any:
    out = {}
    if len(strings) % pair_nargs != 0:
        raise ValueError('incomplete set of key value pairs!')
    index = 0
    for _ in range(len(strings) // pair_nargs):
        assert isinstance(key_nargs, int)
        assert isinstance(val_nargs, int)
        k = strings[index:index + key_nargs]
        index += key_nargs
        v = strings[index:index + val_nargs]
        index += val_nargs
        if key_meta.choices is not None and any((kj not in key_meta.choices for kj in k)):
            raise ValueError(f'invalid choice: {k} (choose from {key_meta.choices}))')
        if val_meta.choices is not None and any((vj not in val_meta.choices for vj in v)):
            raise ValueError(f'invalid choice: {v} (choose from {val_meta.choices}))')
        out[key_instantiator(k)] = val_instantiator(v)
    return out