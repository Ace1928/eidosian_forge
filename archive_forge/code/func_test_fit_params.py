import numpy as np
import numpy.testing as nptest
from numpy.testing import assert_equal
import pytest
from scipy import stats
import statsmodels.api as sm
from statsmodels.graphics import gofplots
from statsmodels.graphics.gofplots import (
from statsmodels.graphics.utils import _import_mpl
def test_fit_params(self):
    assert self.prbplt.fit_params[-2] == self.prbplt.loc
    assert self.prbplt.fit_params[-1] == self.prbplt.scale