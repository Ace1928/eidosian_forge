import os
import warnings
from statsmodels.compat.platform import PLATFORM_WIN
import numpy as np
import pandas as pd
import pytest
from statsmodels.tsa.statespace import sarimax, tools
from .results import results_sarimax
from statsmodels.tools import add_constant
from statsmodels.tools.tools import Bunch
from numpy.testing import (
@pytest.mark.smoke
def test_datasets():
    np.random.seed(232849)
    endog = np.random.binomial(1, 0.5, size=100)
    exog = np.random.binomial(1, 0.5, size=100)
    mod = sarimax.SARIMAX(endog, exog=exog, order=(1, 0, 0))
    mod.fit(disp=-1)