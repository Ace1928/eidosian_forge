import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import scipy.stats
import pytest
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.descriptivestats import (
def test_large_ntop(df):
    res = Description(df, ntop=15)
    assert 'top_15' in res.frame.index