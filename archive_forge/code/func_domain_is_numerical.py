from __future__ import annotations
import sys
import typing
from abc import ABC, abstractmethod
from datetime import MAXYEAR, MINYEAR, datetime, timedelta
from types import MethodType
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
from ._core.dates import datetime_to_num, num_to_datetime
from .breaks import (
from .labels import (
from .utils import identity
@property
def domain_is_numerical(self) -> bool:
    """
        Return True if transformation acts on numerical data.
        e.g. int, float, and imag are numerical but datetime
        is not.

        """
    return isinstance(self.domain[0], (int, float, np.number))