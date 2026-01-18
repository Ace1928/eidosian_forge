from __future__ import annotations
import math
import sys
from datetime import datetime, timezone
from typing import TYPE_CHECKING, cast, overload
from warnings import warn
import numpy as np
import pandas as pd
import pandas.api.types as pdtypes
def round_any(x: FloatArrayLike | float, accuracy: float, f: NumericUFunction=np.round) -> NDArrayFloat | float:
    """
    Round to multiple of any number.
    """
    if not is_vector(x):
        x = np.asarray(x)
    return f(x / accuracy) * accuracy