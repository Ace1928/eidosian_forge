from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
@property
def env_begin(self) -> str:
    return f'\\begin{{tabular}}{{{self.column_format}}}'