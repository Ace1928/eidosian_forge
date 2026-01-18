import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import scipy.stats
import pytest
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.descriptivestats import (
def test_description_basic(df):
    res = Description(df)
    assert isinstance(res.frame, pd.DataFrame)
    assert isinstance(res.numeric, pd.DataFrame)
    assert isinstance(res.categorical, pd.DataFrame)
    assert isinstance(res.summary(), SimpleTable)
    assert isinstance(res.summary().as_text(), str)
    assert 'Descriptive' in str(res)
    res = Description(df.a)
    assert isinstance(res.frame, pd.DataFrame)
    assert isinstance(res.numeric, pd.DataFrame)
    assert isinstance(res.categorical, pd.DataFrame)
    assert isinstance(res.summary(), SimpleTable)
    assert isinstance(res.summary().as_text(), str)
    assert 'Descriptive' in str(res)
    res = Description(df.b)
    assert isinstance(res.frame, pd.DataFrame)
    assert isinstance(res.numeric, pd.DataFrame)
    assert isinstance(res.categorical, pd.DataFrame)
    assert isinstance(res.summary(), SimpleTable)
    assert isinstance(res.summary().as_text(), str)
    assert 'Descriptive' in str(res)