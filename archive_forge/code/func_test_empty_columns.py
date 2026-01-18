import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import scipy.stats
import pytest
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.descriptivestats import (
def test_empty_columns(df):
    df['c'] = np.nan
    res = Description(df)
    dropped = res.frame.c.dropna()
    assert dropped.shape[0] == 2
    assert 'missing' in dropped
    assert 'nobs' in dropped
    df['c'] = np.nan
    res = Description(df.c)
    dropped = res.frame.dropna()
    assert dropped.shape[0] == 2