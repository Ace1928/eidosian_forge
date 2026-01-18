import io
import os
import pytest
import numpy as np
from numpy.testing import assert_allclose, assert_equal
import pandas as pd
import patsy
from statsmodels.api import families
from statsmodels.tools.sm_exceptions import (
from statsmodels.othermod.betareg import BetaModel
from .results import results_betareg as resultsb
def test_income_precision(self):
    rslt = self.income_fit
    assert_close(np.exp(rslt.params[-1:]), expected_income_precision['Estimate'], 0.001)
    assert_close(rslt.pvalues[-1:], expected_income_precision['Pr(>|z|)'], 0.001)