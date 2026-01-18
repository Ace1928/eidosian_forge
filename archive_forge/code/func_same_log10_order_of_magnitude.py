from __future__ import annotations
import math
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast, overload
from warnings import warn
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
def same_log10_order_of_magnitude(x, delta=0.1):
    """
    Return true if range is approximately in same order of magnitude

    For example these sequences are in the same order of magnitude:

        - [1, 8, 5]     # [1, 10)
        - [35, 20, 80]  # [10 100)
        - [232, 730]    # [100, 1000)

    Parameters
    ----------
    x : array-like
         Values in base 10. Must be size 2 and
        ``rng[0] <= rng[1]``.
    delta : float
        Fuzz factor for approximation. It is multiplicative.
    """
    dmin = np.log10(np.min(x) * (1 - delta))
    dmax = np.log10(np.max(x) * (1 + delta))
    return np.floor(dmin) == np.floor(dmax)