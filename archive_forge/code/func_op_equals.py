from __future__ import annotations
from .. import mparser
from .exceptions import InvalidCode, InvalidArguments
from .helpers import flatten, resolve_second_level_holders
from .operator import MesonOperator
from ..mesonlib import HoldableObject, MesonBugException
import textwrap
import typing as T
from abc import ABCMeta
from contextlib import AbstractContextManager
def op_equals(self, other: TYPE_var) -> bool:
    if type(self.held_object) is not type(other):
        self._throw_comp_exception(other, '==')
    return self.held_object == other