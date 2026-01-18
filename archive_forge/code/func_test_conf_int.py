import numpy as np
import pandas as pd
import os
import pytest
from numpy.testing import assert_, assert_equal, assert_allclose
from statsmodels.tsa.statespace.exponential_smoothing import (
def test_conf_int(self):
    ci_95 = self.forecast.conf_int(alpha=0.05)
    lower = results_predict['%s_lower' % self.name]
    upper = results_predict['%s_upper' % self.name]
    assert_allclose(ci_95["lower ('oil', 'data')"], lower.iloc[self.nobs:])
    assert_allclose(ci_95["upper ('oil', 'data')"], upper.iloc[self.nobs:])