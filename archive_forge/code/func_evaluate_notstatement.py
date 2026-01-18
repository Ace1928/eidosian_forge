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
def evaluate_notstatement(self, cur: mparser.NotNode) -> InterpreterObject:
    v = self.evaluate_statement(cur.value)
    if v is None:
        raise InvalidCodeOnVoid('not')
    if isinstance(v, Disabler):
        return v
    return self._holderify(v.operator_call(MesonOperator.NOT, None))