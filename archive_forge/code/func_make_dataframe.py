from typing import Optional
import numpy as np
from packaging.version import Version, parse
import pandas as pd
from pandas.util._decorators import (
def make_dataframe():
    """
        Simple verion of pandas._testing.makeDataFrame
        """
    n = 30
    k = 4
    index = pd.Index(rands_array(nchars=10, size=n), name=None)
    data = {c: pd.Series(np.random.randn(n), index=index) for c in string.ascii_uppercase[:k]}
    return pd.DataFrame(data)