from __future__ import annotations
import re
from functools import total_ordering
from packaging.version import Version
import numpy as np
import pandas as pd
from numba import jit
from pandas.api.extensions import (
from numbers import Integral
from pandas.api.types import pandas_dtype, is_extension_array_dtype
@property
def start_indices(self):
    """
        unsigned integer numpy array the same length as the ragged array where
        values represent the index into flat_array where the corresponding
        ragged array element begins

        Returns
        -------
        np.ndarray
        """
    return self._start_indices