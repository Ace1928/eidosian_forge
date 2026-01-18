from __future__ import annotations
from statsmodels.compat.python import lrange
import warnings
import numpy as np
import pandas as pd
from pandas import DataFrame
from pandas.tseries import offsets
from pandas.tseries.frequencies import to_offset
from typing import Literal
from statsmodels.tools.data import _is_recarray, _is_using_pandas
from statsmodels.tools.sm_exceptions import ValueWarning
from statsmodels.tools.typing import NDArray
from statsmodels.tools.validation import (
def safe_is_const(s):
    try:
        return np.ptp(s) == 0.0 and np.any(s != 0.0)
    except:
        return False