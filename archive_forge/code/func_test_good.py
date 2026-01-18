import numpy as np
import numpy.testing as nptest
from numpy.testing import assert_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics import gofplots
from statsmodels.graphics.gofplots import (
from statsmodels.graphics.utils import _import_mpl
def test_good(self):
    gofplots._check_for(stats.norm, 'ppf')
    gofplots._check_for(stats.norm, 'cdf')