from __future__ import annotations
import sys
import typing
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import pandas as pd
from mizani._core.dates import (
from .utils import NANOSECONDS, SECONDS, log, min_max
def density_max(self, k: float) -> float:
    if k >= self.n:
        return 2 - (k - 1.0) / (self.n - 1.0)
    else:
        return 1