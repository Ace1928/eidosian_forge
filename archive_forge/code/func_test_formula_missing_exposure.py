from statsmodels.compat.pandas import assert_index_equal
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import nbinom
import statsmodels.api as sm
from statsmodels.discrete.discrete_margins import _iscount, _isdummy
from statsmodels.discrete.discrete_model import (
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import (
from .results.results_discrete import Anes, DiscreteL1, RandHIE, Spector
def test_formula_missing_exposure():
    d = {'Foo': [1, 2, 10, 149], 'Bar': [1, 2, 3, np.nan], 'constant': [1] * 4, 'exposure': np.random.uniform(size=4), 'x': [1, 3, 2, 1.5]}
    df = pd.DataFrame(d)
    mod1 = smf.poisson('Foo ~ Bar', data=df, exposure=df['exposure'])
    assert_(type(mod1.exposure) is np.ndarray, msg='Exposure is not ndarray')
    exposure = pd.Series(np.random.uniform(size=5))
    df.loc[3, 'Bar'] = 4
    assert_raises(ValueError, sm.Poisson, df.Foo, df[['constant', 'Bar']], exposure=exposure)