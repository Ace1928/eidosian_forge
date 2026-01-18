from __future__ import annotations
import contextlib
import string
import warnings
import numpy as np
import pandas as pd
from packaging.version import Version
import pandas.testing as tm
def makeMixedDataFrame():
    df = pd.DataFrame({'A': [0.0, 1, 2, 3, 4], 'B': [0.0, 1, 0, 1, 0], 'C': [f'foo{i}' for i in range(5)], 'D': pd.date_range('2009-01-01', periods=5)})
    return df