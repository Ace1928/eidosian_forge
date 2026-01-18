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
@FeatureNew('multiline format strings', '0.63.0')
def evaluate_multiline_fstring(self, node: mparser.MultilineFormatStringNode) -> InterpreterObject:
    return self.evaluate_fstring(node)