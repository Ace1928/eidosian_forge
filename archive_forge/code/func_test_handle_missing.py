from statsmodels.compat.platform import PLATFORM_OSX
import os
import csv
import warnings
import numpy as np
import pandas as pd
from scipy import sparse
import pytest
from statsmodels.regression.mixed_linear_model import (
from numpy.testing import (assert_almost_equal, assert_equal, assert_allclose,
from statsmodels.base import _penalties as penalties
import statsmodels.tools.numdiff as nd
from .results import lme_r_results
@pytest.mark.slow
def test_handle_missing():
    np.random.seed(23423)
    df = np.random.normal(size=(100, 6))
    df = pd.DataFrame(df)
    df.columns = ['y', 'g', 'x1', 'z1', 'c1', 'c2']
    df['g'] = np.kron(np.arange(50), np.ones(2))
    re = np.random.normal(size=(50, 4))
    re = np.kron(re, np.ones((2, 1)))
    df['y'] = re[:, 0] + re[:, 1] * df.z1 + re[:, 2] * df.c1
    df['y'] += re[:, 3] * df.c2 + np.random.normal(size=100)
    df.loc[1, 'y'] = np.nan
    df.loc[2, 'g'] = np.nan
    df.loc[3, 'x1'] = np.nan
    df.loc[4, 'z1'] = np.nan
    df.loc[5, 'c1'] = np.nan
    df.loc[6, 'c2'] = np.nan
    fml = 'y ~ x1'
    re_formula = '1 + z1'
    vc_formula = {'a': '0 + c1', 'b': '0 + c2'}
    for include_re in (False, True):
        for include_vc in (False, True):
            kwargs = {}
            dx = df.copy()
            va = ['y', 'g', 'x1']
            if include_re:
                kwargs['re_formula'] = re_formula
                va.append('z1')
            if include_vc:
                kwargs['vc_formula'] = vc_formula
                va.extend(['c1', 'c2'])
            dx = dx[va].dropna()
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                model1 = MixedLM.from_formula(fml, groups='g', data=dx, **kwargs)
                result1 = model1.fit()
                model2 = MixedLM.from_formula(fml, groups='g', data=df, missing='drop', **kwargs)
                result2 = model2.fit()
                assert_allclose(result1.params, result2.params)
                assert_allclose(result1.bse, result2.bse)
                assert_equal(len(result1.fittedvalues), result1.nobs)