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
def test_negative_binomial_default_alpha_param():
    with pytest.warns(UserWarning, match='Negative binomial dispersion parameter alpha not set'):
        sm.families.NegativeBinomial()
    with pytest.warns(UserWarning, match='Negative binomial dispersion parameter alpha not set'):
        sm.families.NegativeBinomial(link=sm.families.links.nbinom(alpha=1.0))
    with warnings.catch_warnings():
        warnings.simplefilter('error')
        sm.families.NegativeBinomial(alpha=1.0)
    with pytest.warns(FutureWarning):
        sm.families.NegativeBinomial(link=sm.families.links.nbinom(alpha=1.0), alpha=1.0)