import numpy as np
from numpy.testing import assert_almost_equal, assert_equal
import pandas as pd
import scipy.stats
import pytest
from statsmodels.iolib.table import SimpleTable
from statsmodels.stats.descriptivestats import (
@pytest.mark.skipif(not hasattr(pd, 'NA'), reason='Must support NA')
def test_extension_types(df):
    df['c'] = pd.Series(np.arange(100.0))
    df['d'] = pd.Series(np.arange(100), dtype=pd.Int64Dtype())
    df.loc[df.index[::2], 'c'] = np.nan
    df.loc[df.index[::2], 'd'] = pd.NA
    res = Description(df)
    np.testing.assert_allclose(res.frame.c, res.frame.d)