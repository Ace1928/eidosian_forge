import numpy as np
import pytest
from statsmodels import datasets
from statsmodels.tsa.statespace import (
from numpy.testing import assert_, assert_raises, assert_equal, assert_allclose
def test_mle_validate():
    mod = mlemodel.MLEModel([], 1)
    mod._param_names = ['a', 'b', 'c']
    msg = 'Invalid parameter name passed: "d"'
    with pytest.raises(ValueError, match=msg):
        with mod.fix_params({'d': 1}):
            pass