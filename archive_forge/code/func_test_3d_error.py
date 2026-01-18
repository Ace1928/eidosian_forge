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
def test_3d_error(self):
    data = np.array(2)
    with pytest.raises(ValueError):
        stattools.lagmat2ds(data, 5)
    data = np.zeros((100, 2, 2))
    with pytest.raises(ValueError):
        stattools.lagmat2ds(data, 5)