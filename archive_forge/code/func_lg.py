import numpy as np
from numpy.testing import assert_array_less, assert_equal, assert_raises
from pandas import DataFrame, Series
import pytest
import statsmodels.api as sm
from statsmodels.graphics.regressionplots import (
@np.vectorize
def lg(x):
    return np.log10(x) if x > 0 else 0