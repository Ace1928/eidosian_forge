from __future__ import annotations
import re
import os
import typing as T
from ...mesonlib import version_compare
from ...interpreterbase import (
@FeatureNew('"not in" string operator', '1.0.0')
@typed_operator(MesonOperator.NOT_IN, str)
def op_notin(self, other: str) -> bool:
    return other not in self.held_object