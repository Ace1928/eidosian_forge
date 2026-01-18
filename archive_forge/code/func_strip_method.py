from __future__ import annotations
import re
import os
import typing as T
from ...mesonlib import version_compare
from ...interpreterbase import (
@noKwargs
@typed_pos_args('str.strip', optargs=[str])
def strip_method(self, args: T.Tuple[T.Optional[str]], kwargs: TYPE_kwargs) -> str:
    if args[0]:
        FeatureNew.single_use('str.strip with a positional argument', '0.43.0', self.subproject, location=self.current_node)
    return self.held_object.strip(args[0])