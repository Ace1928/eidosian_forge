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
def evaluate_plusassign(self, node: mparser.PlusAssignmentNode) -> None:
    assert isinstance(node, mparser.PlusAssignmentNode)
    varname = node.var_name.value
    addition = self.evaluate_statement(node.value)
    if addition is None:
        raise InvalidCodeOnVoid('plus assign')
    old_variable = self.get_variable(varname)
    old_variable.current_node = node
    new_value = self._holderify(old_variable.operator_call(MesonOperator.PLUS, _unholder(addition)))
    self.set_variable(varname, new_value)