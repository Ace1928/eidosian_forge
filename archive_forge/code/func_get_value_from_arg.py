from __future__ import annotations
import dataclasses
import itertools
from typing import Any, Callable, Dict, List, Sequence, Set, Tuple, TypeVar, Union
from typing_extensions import get_args
from . import _arguments, _fields, _parsers, _resolver, _strings
from .conf import _markers
def get_value_from_arg(prefixed_field_name: str) -> Any:
    """Helper for getting values from `value_from_arg` + doing some extra
        asserts."""
    assert prefixed_field_name in value_from_prefixed_field_name, f'{prefixed_field_name} not in {value_from_prefixed_field_name}'
    return value_from_prefixed_field_name[prefixed_field_name]