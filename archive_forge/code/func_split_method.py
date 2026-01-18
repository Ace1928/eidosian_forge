from __future__ import annotations
import re
import os
import typing as T
from ...mesonlib import version_compare
from ...interpreterbase import (
@noKwargs
@typed_pos_args('str.split', optargs=[str])
def split_method(self, args: T.Tuple[T.Optional[str]], kwargs: TYPE_kwargs) -> T.List[str]:
    return self.held_object.split(args[0])