from __future__ import annotations
import re
import os
import typing as T
from ...mesonlib import version_compare
from ...interpreterbase import (
@noKwargs
@FeatureNew('str.substring', '0.56.0')
@typed_pos_args('str.substring', optargs=[int, int])
def substring_method(self, args: T.Tuple[T.Optional[int], T.Optional[int]], kwargs: TYPE_kwargs) -> str:
    start = args[0] if args[0] is not None else 0
    end = args[1] if args[1] is not None else len(self.held_object)
    return self.held_object[start:end]