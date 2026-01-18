import os
import numpy as np
from numpy.testing import assert_allclose, assert_equal, assert_raises
import pandas as pd
import pytest
from scipy.stats import norm
from statsmodels.datasets import macrodata
from statsmodels.genmod.api import GLM
from statsmodels.regression.linear_model import OLS
from statsmodels.regression.recursive_ls import RecursiveLS
from statsmodels.stats.diagnostic import recursive_olsresiduals
from statsmodels.tools import add_constant
from statsmodels.tools.eval_measures import aic, bic
from statsmodels.tools.sm_exceptions import ValueWarning
def test_glm_constrained():
    test_glm(constraints='m1 + unemp = 1')