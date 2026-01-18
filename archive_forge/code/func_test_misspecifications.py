import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_allclose, assert_raises
import pandas as pd
import pytest
from statsmodels.tsa.statespace import varmax, sarimax
from statsmodels.iolib.summary import forg
from .results import results_varmax
def test_misspecifications():
    varmax.__warningregistry__ = {}
    endog = np.arange(20).reshape(10, 2)
    with pytest.raises(ValueError):
        varmax.VARMAX(endog, order=(1, 0), trend='')
    with pytest.raises(ValueError):
        varmax.VARMAX(endog, order=(1, 0), error_cov_type='')
    with pytest.raises(ValueError):
        varmax.VARMAX(endog, order=(0, 0))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        varmax.VARMAX(endog, order=(1, 1))
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter('always')
        varmax.VARMAX(endog, order=(1, 1))
        message = 'Estimation of VARMA(p,q) models is not generically robust, due especially to identification issues.'
        assert str(w[0].message) == message
    warnings.resetwarnings()