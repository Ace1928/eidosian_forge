from io import BytesIO
from itertools import product
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from statsmodels import tools
from statsmodels.regression.linear_model import WLS
from statsmodels.regression.rolling import RollingWLS, RollingOLS
def get_sub(x, idx, window):
    if isinstance(x, (pd.Series, pd.DataFrame)):
        out = x.iloc[idx - window:idx]
        return np.asarray(out)
    return x[idx - window:idx]