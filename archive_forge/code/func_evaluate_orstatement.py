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
def evaluate_orstatement(self, cur: mparser.OrNode) -> InterpreterObject:
    l = self.evaluate_statement(cur.left)
    if l is None:
        raise mesonlib.MesonException('Cannot compare a void statement on the left-hand side')
    if isinstance(l, Disabler):
        return l
    l_bool = l.operator_call(MesonOperator.BOOL, None)
    if l_bool:
        return self._holderify(l_bool)
    r = self.evaluate_statement(cur.right)
    if r is None:
        raise mesonlib.MesonException('Cannot compare a void statement on the right-hand side')
    if isinstance(r, Disabler):
        return r
    return self._holderify(r.operator_call(MesonOperator.BOOL, None))