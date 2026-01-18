from __future__ import annotations
import sys
import typing
from datetime import datetime, timedelta
from itertools import product
import numpy as np
import pandas as pd
from mizani._core.dates import (
from .utils import NANOSECONDS, SECONDS, log, min_max
@classmethod
def format_info(cls, x: NDArrayTimedelta, units: Optional[DurationUnit]=None) -> tuple[NDArrayFloat, DurationUnit]:
    helper = cls(x, units)
    return (helper.timedelta_to_numeric(x), helper.units)