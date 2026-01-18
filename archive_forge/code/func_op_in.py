from __future__ import annotations
import re
import os
import typing as T
from ...mesonlib import version_compare
from ...interpreterbase import (
@FeatureNew('"in" string operator', '1.0.0')
@typed_operator(MesonOperator.IN, str)
def op_in(self, other: str) -> bool:
    return other in self.held_object