from __future__ import annotations
import typing
from copy import copy
import numpy as np
import pandas as pd
from .._utils import groupby_apply, pivot_apply
from ..exceptions import PlotnineError
from .position_dodge import position_dodge
def find_x_overlaps(df: pd.DataFrame) -> IntArray:
    """
    Find overlapping regions along the x axis
    """
    n = len(df)
    overlaps = np.zeros(n, dtype=int)
    overlaps[0] = 1
    counter = 1
    for i in range(1, n):
        if df['xmin'].iloc[i] >= df['xmax'].iloc[i - 1]:
            counter += 1
        overlaps[i] = counter
    return overlaps