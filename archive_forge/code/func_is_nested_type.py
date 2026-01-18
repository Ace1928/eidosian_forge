from __future__ import annotations
import collections
import collections.abc
import dataclasses
import enum
import functools
import inspect
import itertools
import numbers
import os
import sys
import typing
import warnings
from typing import (
import docstring_parser
import typing_extensions
from typing_extensions import (
from . import (
from ._typing import TypeForm
from .conf import _confstruct, _markers
@_unsafe_cache.unsafe_cache(maxsize=1024)
def is_nested_type(typ: Union[TypeForm[Any], Callable], default_instance: DefaultInstance) -> bool:
    """Determine whether a type should be treated as a 'nested type', where a single
    type can be broken down into multiple fields (eg for nested dataclasses or
    classes).

    TODO: we should come up with a better name than 'nested type', which is a little bit
    misleading."""
    _, argconfs = _resolver.unwrap_annotated(typ, search_type=conf._confstruct._ArgConfiguration)
    _, subcommand_confs = _resolver.unwrap_annotated(typ, search_type=conf._confstruct._SubcommandConfiguration)
    for union_conf in argconfs + subcommand_confs:
        if union_conf.constructor_factory is not None:
            typ = union_conf.constructor_factory()
    return not isinstance(_try_field_list_from_callable(typ, default_instance), UnsupportedNestedTypeMessage)