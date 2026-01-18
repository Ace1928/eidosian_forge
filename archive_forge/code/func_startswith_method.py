from __future__ import annotations
import re
import os
import typing as T
from ...mesonlib import version_compare
from ...interpreterbase import (
@noKwargs
@typed_pos_args('str.startswith', str)
def startswith_method(self, args: T.Tuple[str], kwargs: TYPE_kwargs) -> bool:
    return self.held_object.startswith(args[0])