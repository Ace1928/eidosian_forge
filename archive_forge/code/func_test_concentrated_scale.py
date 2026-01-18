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
@pytest.mark.slow
def test_concentrated_scale():
    check_concentrated_scale(filter_univariate=False)
    check_concentrated_scale(filter_univariate=True)