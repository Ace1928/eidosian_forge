from __future__ import annotations
import contextlib
import string
import warnings
import numpy as np
import pandas as pd
from packaging.version import Version
import pandas.testing as tm
def makeDateIndex(k=30, freq='B'):
    return pd.date_range('2000', periods=k, freq=freq)