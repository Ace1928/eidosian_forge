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
def evaluate_foreach(self, node: mparser.ForeachClauseNode) -> None:
    assert isinstance(node, mparser.ForeachClauseNode)
    items = self.evaluate_statement(node.items)
    if not isinstance(items, IterableObject):
        raise InvalidArguments('Items of foreach loop do not support iterating')
    tsize = items.iter_tuple_size()
    if len(node.varnames) != (tsize or 1):
        raise InvalidArguments(f'Foreach expects exactly {tsize or 1} variables for iterating over objects of type {items.display_name()}')
    for i in items.iter_self():
        if tsize is None:
            if isinstance(i, tuple):
                raise mesonlib.MesonBugException(f'Iteration of {items} returned a tuple even though iter_tuple_size() is None')
            self.set_variable(node.varnames[0].value, self._holderify(i))
        else:
            if not isinstance(i, tuple):
                raise mesonlib.MesonBugException(f'Iteration of {items} did not return a tuple even though iter_tuple_size() is {tsize}')
            if len(i) != tsize:
                raise mesonlib.MesonBugException(f'Iteration of {items} did not return a tuple even though iter_tuple_size() is {tsize}')
            for j in range(tsize):
                self.set_variable(node.varnames[j].value, self._holderify(i[j]))
        try:
            self.evaluate_codeblock(node.block)
        except ContinueRequest:
            continue
        except BreakRequest:
            break