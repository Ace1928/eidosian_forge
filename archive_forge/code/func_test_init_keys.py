from statsmodels.compat.platform import PLATFORM_LINUX32
import numpy as np
from numpy.testing import (
import pandas as pd
import pytest
import statsmodels.api as sm
from .results.results_discrete import RandHIE
from .test_discrete import CheckModelMixin
def test_init_keys(self):
    init_kwds = self.res1.model._get_init_kwds()
    assert_equal(set(init_kwds.keys()), set(self.init_keys))
    for key, value in self.init_kwds.items():
        assert_equal(init_kwds[key], value)