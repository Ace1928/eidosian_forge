import numpy as np
import pandas as pd
import pytest
from statsmodels.imputation import mice
import statsmodels.api as sm
from numpy.testing import assert_equal, assert_allclose
import warnings
def gendat():
    """
    Create a data set with missing values.
    """
    gen = np.random.RandomState(34243)
    n = 200
    p = 5
    exog = gen.normal(size=(n, p))
    exog[:, 0] = exog[:, 1] - exog[:, 2] + 2 * exog[:, 4]
    exog[:, 0] += gen.normal(size=n)
    exog[:, 2] = 1 * (exog[:, 2] > 0)
    endog = exog.sum(1) + gen.normal(size=n)
    df = pd.DataFrame(exog)
    df.columns = ['x%d' % k for k in range(1, p + 1)]
    df['y'] = endog
    df.loc[0:59, 'x1'] = np.nan
    df.loc[0:39, 'x2'] = np.nan
    df.loc[10:29:2, 'x3'] = np.nan
    df.loc[20:49:3, 'x4'] = np.nan
    df.loc[40:44, 'x5'] = np.nan
    df.loc[30:99:2, 'y'] = np.nan
    return df