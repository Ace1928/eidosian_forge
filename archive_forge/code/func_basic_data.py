from io import BytesIO
from itertools import product
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_allclose, assert_array_equal
from statsmodels import tools
from statsmodels.regression.linear_model import WLS
from statsmodels.regression.rolling import RollingWLS, RollingOLS
@pytest.fixture(scope='module', params=basic_params, ids=ids)
def basic_data(request):
    return gen_data(*request.param)