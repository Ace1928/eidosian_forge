from statsmodels.compat.pandas import QUARTER_END
import numpy as np
from numpy.testing import assert_allclose
import pandas as pd
import pytest
from scipy.signal import lfilter
from statsmodels.tsa.statespace import (
def test_em_nonstationary(reset_randomstate):
    ix = pd.period_range(start='2000', periods=20, freq='M')
    endog_M = pd.Series(np.arange(20), index=ix, dtype=float)
    endog_M.iloc[10:12] += [0.4, -0.2]
    ix = pd.period_range(start='2000', periods=5, freq='Q')
    endog_Q = pd.Series(np.arange(5), index=ix)
    mod = dynamic_factor_mq.DynamicFactorMQ(endog_M, endog_quarterly=endog_Q, idiosyncratic_ar1=False, standardize=False, factors=['global'])
    msg = 'Non-stationary parameters found at EM iteration 1, which is not compatible with stationary initialization. Initialization was switched to diffuse for the following:  \\["factor block: \\(\\\'global\\\',\\)"\\], and fitting was restarted.'
    with pytest.warns(UserWarning, match=msg):
        mod.fit(maxiter=2, em_initialization=False)