from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
def test_nanvar_axis(self, samples, variance):
    samples_unif = self.prng.uniform(size=samples.shape[0])
    samples = np.vstack([samples, samples_unif])
    actual_variance = nanops.nanvar(samples, axis=1)
    tm.assert_almost_equal(actual_variance, np.array([variance, 1.0 / 12]), rtol=0.01)