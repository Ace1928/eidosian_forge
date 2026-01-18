from __future__ import annotations
import re
import os
import typing as T
from ...mesonlib import version_compare
from ...interpreterbase import (
@noKwargs
@typed_pos_args('str.join', varargs=str)
def join_method(self, args: T.Tuple[T.List[str]], kwargs: TYPE_kwargs) -> str:
    return self.held_object.join(args[0])