from __future__ import annotations
import sys
import typing
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import pandas as pd
from mizani._core.dates import (
from .utils import NANOSECONDS, SECONDS, log, min_max
def simplicity(self, q: float, j: float, lmin: float, lmax: float, lstep: float) -> float:
    eps = 1e-10
    n = len(self.Q)
    i = self.Q_index[q] + 1
    if (lmin % lstep < eps or lstep - lmin % lstep < eps) and lmin <= 0 and (lmax >= 0):
        v = 1
    else:
        v = 0
    return (n - i) / (n - 1.0) + v - j