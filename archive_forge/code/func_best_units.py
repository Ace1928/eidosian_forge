from __future__ import annotations
import sys
import typing
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import pandas as pd
from mizani._core.dates import (
from .utils import NANOSECONDS, SECONDS, log, min_max
def best_units(self, x: NDArrayTimedelta | Sequence[Timedelta]) -> DurationUnit:
    """
        Determine good units for representing a sequence of timedeltas
        """
    ts_range = self.value(max(x)) - self.value(min(x))
    package = self.determine_package(x[0])
    if package == 'pandas':
        cuts: list[tuple[float, DurationUnit]] = [(0.9, 'us'), (0.9, 'ms'), (0.9, 's'), (9, 'min'), (6, 'h'), (4, 'day'), (4, 'week'), (4, 'month'), (3, 'year')]
        denomination = NANOSECONDS
        base_units = 'ns'
    else:
        cuts = [(0.9, 's'), (9, 'min'), (6, 'h'), (4, 'day'), (4, 'week'), (4, 'month'), (3, 'year')]
        denomination = SECONDS
        base_units = 'ms'
    for size, units in reversed(cuts):
        if ts_range >= size * denomination[units]:
            return units
    return base_units