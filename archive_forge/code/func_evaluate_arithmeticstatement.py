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
def evaluate_arithmeticstatement(self, cur: mparser.ArithmeticNode) -> InterpreterObject:
    l = self.evaluate_statement(cur.left)
    if isinstance(l, Disabler):
        return l
    r = self.evaluate_statement(cur.right)
    if isinstance(r, Disabler):
        return r
    if l is None or r is None:
        raise InvalidCodeOnVoid(cur.operation)
    mapping: T.Dict[str, MesonOperator] = {'add': MesonOperator.PLUS, 'sub': MesonOperator.MINUS, 'mul': MesonOperator.TIMES, 'div': MesonOperator.DIV, 'mod': MesonOperator.MOD}
    l.current_node = cur
    res = l.operator_call(mapping[cur.operation], _unholder(r))
    return self._holderify(res)