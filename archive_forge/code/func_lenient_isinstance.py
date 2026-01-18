from __future__ import annotations as _annotations
import dataclasses
import keyword
import typing
import weakref
from collections import OrderedDict, defaultdict, deque
from copy import deepcopy
from itertools import zip_longest
from types import BuiltinFunctionType, CodeType, FunctionType, GeneratorType, LambdaType, ModuleType
from typing import Any, Mapping, TypeVar
from typing_extensions import TypeAlias, TypeGuard
from . import _repr, _typing_extra
def lenient_isinstance(o: Any, class_or_tuple: type[Any] | tuple[type[Any], ...] | None) -> bool:
    try:
        return isinstance(o, class_or_tuple)
    except TypeError:
        return False