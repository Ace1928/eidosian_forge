import os
import re
import warnings
import numpy as np
from numpy.testing import assert_equal, assert_raises, assert_allclose
import pandas as pd
import pytest
from statsmodels.tsa.statespace import dynamic_factor
from .results import results_varmax, results_dynamic_factor
from statsmodels.iolib.summary import forg
def test_bse_approx(self):
    bse = self.results._cov_params_approx().diagonal()
    assert_allclose(bse, self.true['var_oim'], atol=1e-05)