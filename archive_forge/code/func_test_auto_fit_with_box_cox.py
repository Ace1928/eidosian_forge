from pathlib import Path
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import pytest
from statsmodels.tsa.seasonal import MSTL
def test_auto_fit_with_box_cox(data):
    periods = (5, 6, 7)
    mod = MSTL(endog=data, periods=periods, lmbda='auto')
    mod.fit()
    assert hasattr(mod, 'est_lmbda')
    assert isinstance(mod.est_lmbda, float)