import warnings
import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.robust import norms
from statsmodels.robust.robust_linear_model import RLM
from statsmodels.robust.scale import HuberScale
def load_stackloss():
    from statsmodels.datasets.stackloss import load
    data = load()
    data.endog = np.asarray(data.endog)
    data.exog = np.asarray(data.exog)
    return data