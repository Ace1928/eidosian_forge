from __future__ import annotations
from abc import (
from typing import (
import numpy as np
from pandas.core.dtypes.generic import ABCMultiIndex
def pad_empties(x):
    for pad in reversed(x):
        if pad:
            return [x[0]] + [i if i else ' ' * len(pad) for i in x[1:]]