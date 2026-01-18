from statsmodels.compat.platform import PLATFORM_OSX
from statsmodels.regression.process_regression import (
import numpy as np
import pandas as pd
import pytest
import collections
import statsmodels.tools.numdiff as nd
from numpy.testing import assert_allclose, assert_equal
def run_arrays(n, get_model, noise):
    y, x_mean, x_sc, x_sm, x_no, time, groups = setup1(n, get_model, noise)
    preg = ProcessMLE(y, x_mean, x_sc, x_sm, x_no, time, groups)
    return preg.fit()