from statsmodels.compat.pandas import assert_frame_equal, assert_series_equal
from statsmodels.compat.python import lrange
import string
import numpy as np
from numpy.random import standard_normal
from numpy.testing import (
import pandas as pd
import pytest
from statsmodels.datasets import longley
from statsmodels.tools import tools
from statsmodels.tools.tools import pinv_extended
@pytest.fixture(scope='module')
def string_var():
    string_var = [string.ascii_lowercase[0:5], string.ascii_lowercase[5:10], string.ascii_lowercase[10:15], string.ascii_lowercase[15:20], string.ascii_lowercase[20:25]]
    string_var *= 5
    string_var = np.asarray(sorted(string_var))
    series = pd.Series(string_var, name='string_var')
    return series