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
def test_bse_robust(self):
    robust_oim_bse = self.result.cov_params_robust_oim.diagonal() ** 0.5
    cpra = self.result.cov_params_robust_approx
    robust_approx_bse = cpra.diagonal() ** 0.5
    true_robust_bse = np.r_[self.true['se_ar_robust'], self.true['se_ma_robust']]
    assert_allclose(robust_oim_bse[1:3], true_robust_bse, atol=0.01)
    assert_allclose(robust_approx_bse[1:3], true_robust_bse, atol=0.001)