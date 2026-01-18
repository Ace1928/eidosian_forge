from __future__ import annotations
import sys
import typing
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import pandas as pd
from mizani._core.dates import (
from .utils import NANOSECONDS, SECONDS, log, min_max
def simplicity_max(self, q: float, j: float) -> float:
    n = len(self.Q)
    i = self.Q_index[q] + 1
    v = 1
    return (n - i) / (n - 1.0) + v - j