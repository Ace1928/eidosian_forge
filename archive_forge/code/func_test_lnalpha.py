from statsmodels.compat.pandas import assert_index_equal
import os
import warnings
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
from scipy import stats
from scipy.stats import nbinom
import statsmodels.api as sm
from statsmodels.discrete.discrete_margins import _iscount, _isdummy
from statsmodels.discrete.discrete_model import (
import statsmodels.formula.api as smf
from statsmodels.tools.sm_exceptions import (
from .results.results_discrete import Anes, DiscreteL1, RandHIE, Spector
def test_lnalpha(self):
    assert_allclose(self.res1.lnalpha, self.res2.lnalpha, atol=0.001, rtol=0.001)
    assert_allclose(self.res1.lnalpha_std_err, self.res2.lnalpha_std_err, atol=0.001, rtol=0.001)