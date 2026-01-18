from __future__ import annotations
import os
import sys
import typing as T
from .. import mparser, mesonlib
from .. import environment
from ..interpreterbase import (
from ..interpreter import (
from ..mparser import (
def func_do_nothing(self, node: BaseNode, args: T.List[TYPE_var], kwargs: T.Dict[str, TYPE_var]) -> bool:
    return True