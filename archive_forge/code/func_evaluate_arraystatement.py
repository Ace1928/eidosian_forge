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
def evaluate_arraystatement(self, cur: mparser.ArrayNode) -> InterpreterObject:
    arguments, kwargs = self.reduce_arguments(cur.args)
    if len(kwargs) > 0:
        raise InvalidCode('Keyword arguments are invalid in array construction.')
    return self._holderify([_unholder(x) for x in arguments])