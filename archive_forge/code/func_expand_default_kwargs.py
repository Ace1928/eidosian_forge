from __future__ import annotations
from .. import environment, mparser, mesonlib
from .baseobjects import (
from .exceptions import (
from .decorators import FeatureNew
from .disabler import Disabler, is_disabled
from .helpers import default_resolve_key, flatten, resolve_second_level_holders, stringifyUserArguments
from .operator import MesonOperator
from ._unholder import _unholder
import os, copy, re, pathlib
import typing as T
import textwrap
def expand_default_kwargs(self, kwargs: T.Dict[str, T.Optional[InterpreterObject]]) -> T.Dict[str, T.Optional[InterpreterObject]]:
    if 'kwargs' not in kwargs:
        return kwargs
    to_expand = _unholder(kwargs.pop('kwargs'))
    if not isinstance(to_expand, dict):
        raise InterpreterException('Value of "kwargs" must be dictionary.')
    if 'kwargs' in to_expand:
        raise InterpreterException('Kwargs argument must not contain a "kwargs" entry. Points for thinking meta, though. :P')
    for k, v in to_expand.items():
        if k in kwargs:
            raise InterpreterException(f'Entry "{k}" defined both as a keyword argument and in a "kwarg" entry.')
        kwargs[k] = self._holderify(v)
    return kwargs