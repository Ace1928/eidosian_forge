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
def fixed_length_tuple_instantiator(strings: List[str]) -> Any:
    assert len(strings) == nargs
    out = []
    i = 0
    for make, meta in zip(instantiators, metas):
        assert isinstance(meta.nargs, int)
        meta.check_choices(strings[i:i + meta.nargs])
        out.append(make(strings[i:i + meta.nargs]))
        i += meta.nargs
    return tuple(out)