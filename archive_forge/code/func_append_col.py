from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
def append_col() -> None:
    if ncol > 1:
        row2.append(f'\\multicolumn{{{ncol:d}}}{{{self.multicolumn_format}}}{{{coltext.strip()}}}')
    else:
        row2.append(coltext)