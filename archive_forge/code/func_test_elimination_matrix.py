from statsmodels.compat.pandas import (
import numpy as np
from numpy.testing import (
import pandas as pd
from pandas.tseries.frequencies import to_offset
import pytest
from statsmodels import regression
from statsmodels.datasets import macrodata
from statsmodels.tsa import stattools
from statsmodels.tsa.tests.results import savedrvs
from statsmodels.tsa.tests.results.datamlw_tls import (
import statsmodels.tsa.tsatools as tools
from statsmodels.tsa.tsatools import vec, vech
def test_elimination_matrix(reset_randomstate):
    for k in range(2, 10):
        m = np.random.randn(k, k)
        Lk = tools.elimination_matrix(k)
        assert np.array_equal(vech(m), np.dot(Lk, vec(m)))