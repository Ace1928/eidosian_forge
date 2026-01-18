import warnings
import numpy as np
from numpy.testing import assert_almost_equal, assert_allclose
import scipy.sparse as sparse
import pytest
from statsmodels.stats.correlation_tools import (
from statsmodels.tools.testing import Holder
def test_corr_thresholded(self, reset_randomstate):
    import datetime
    t1 = datetime.datetime.now()
    X = np.random.normal(size=(2000, 10))
    tcor = corr_thresholded(X, 0.2, max_elt=4000000.0)
    t2 = datetime.datetime.now()
    ss = (t2 - t1).seconds
    fcor = np.corrcoef(X)
    fcor *= np.abs(fcor) >= 0.2
    assert_allclose(tcor.todense(), fcor, rtol=0.25, atol=0.001)