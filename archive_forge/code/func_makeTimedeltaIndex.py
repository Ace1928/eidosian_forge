from __future__ import annotations
import contextlib
import string
import warnings
import numpy as np
import pandas as pd
from packaging.version import Version
import pandas.testing as tm
def makeTimedeltaIndex(k=30, freq='D'):
    return pd.timedelta_range('1 day', periods=k, freq=freq)