from __future__ import annotations
import math
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast, overload
from warnings import warn
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
def isclose_abs(a: float, b: float, tol: float=ABS_TOL) -> bool:
    """
    Return True if a and b are close given the absolute tolerance
    """
    return math.isclose(a, b, rel_tol=0, abs_tol=ABS_TOL)