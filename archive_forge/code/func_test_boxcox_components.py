from statsmodels.compat.pandas import MONTH_END
from statsmodels.compat.pytest import pytest_warns
import os
import re
import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pandas as pd
import pytest
import scipy.stats
from statsmodels.tools.sm_exceptions import ConvergenceWarning, ValueWarning
from statsmodels.tsa.holtwinters import (
from statsmodels.tsa.holtwinters._exponential_smoothers import (
from statsmodels.tsa.holtwinters._smoothers import (
def test_boxcox_components(ses):
    mod = ExponentialSmoothing(ses + 1 - ses.min(), initialization_method='estimated', use_boxcox=True)
    res = mod.fit()
    assert isinstance(res.summary().as_text(), str)
    with pytest.raises(AssertionError):
        assert_allclose(res.level, res.fittedvalues)
    assert not hasattr(res, '_untransformed_level')
    assert not hasattr(res, '_untransformed_trend')
    assert not hasattr(res, '_untransformed_seasonal')