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
def evaluate_ternary(self, node: mparser.TernaryNode) -> T.Optional[InterpreterObject]:
    assert isinstance(node, mparser.TernaryNode)
    result = self.evaluate_statement(node.condition)
    if result is None:
        raise mesonlib.MesonException('Cannot use a void statement as condition for ternary operator.')
    if isinstance(result, Disabler):
        return result
    result.current_node = node
    result_bool = result.operator_call(MesonOperator.BOOL, None)
    if result_bool:
        return self.evaluate_statement(node.trueblock)
    else:
        return self.evaluate_statement(node.falseblock)