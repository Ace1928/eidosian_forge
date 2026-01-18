from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
def get_col_type(dtype) -> str:
    if issubclass(dtype.type, np.number):
        return 'r'
    return 'l'