from statsmodels.compat.python import asbytes
from io import BytesIO
import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_, assert_allclose, assert_almost_equal, assert_equal, \
from statsmodels.stats.libqsturng import qsturng
from statsmodels.stats.multicomp import (tukeyhsd, pairwise_tukeyhsd,
def test_multicomptukey(self):
    assert_almost_equal(self.res.meandiffs, self.meandiff2, decimal=14)
    assert_almost_equal(self.res.confint, self.confint2, decimal=2)
    assert_equal(self.res.reject, self.reject2)