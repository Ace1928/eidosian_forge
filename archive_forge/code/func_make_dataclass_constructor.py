from __future__ import annotations
import dataclasses
import enum
import functools
from typing import IO, TYPE_CHECKING, Any, Optional, Set, Type, TypeVar, Union
from typing_extensions import get_args, get_origin
from .. import _fields, _resolver
def make_dataclass_constructor(typ: Type[Any]):
    return lambda loader, node: typ(**loader.construct_mapping(node))