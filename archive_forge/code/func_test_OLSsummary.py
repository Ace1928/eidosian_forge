import warnings
import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_equal
from statsmodels.iolib.summary2 import summary_col
from statsmodels.tools.tools import add_constant
from statsmodels.regression.linear_model import OLS
def test_OLSsummary(self):
    x = [1, 5, 7, 3, 5]
    x = add_constant(x)
    y1 = [6, 4, 2, 7, 4]
    reg1 = OLS(y1, x).fit()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        actual = reg1.summary().as_latex()
    string_to_find = '\\end{tabular}\n\\begin{tabular}'
    result = string_to_find in actual
    assert result is True