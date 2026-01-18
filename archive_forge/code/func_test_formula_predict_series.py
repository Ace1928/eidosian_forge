from statsmodels.compat.pandas import assert_series_equal
from io import StringIO
import warnings
import numpy as np
import numpy.testing as npt
import pandas as pd
import patsy
import pytest
from statsmodels.datasets import cpunish
from statsmodels.datasets.longley import load, load_pandas
from statsmodels.formula.api import ols
from statsmodels.formula.formulatools import make_hypotheses_matrices
from statsmodels.tools import add_constant
from statsmodels.tools.testing import assert_equal
def test_formula_predict_series():
    data = pd.DataFrame({'y': [1, 2, 3], 'x': [1, 2, 3]}, index=[5, 3, 1])
    results = ols('y ~ x', data).fit()
    result = results.predict(data)
    expected = pd.Series([1.0, 2.0, 3.0], index=[5, 3, 1])
    assert_series_equal(result, expected)
    result = results.predict(data.x)
    assert_series_equal(result, expected)
    result = results.predict(pd.Series([1, 2, 3], index=[1, 2, 3], name='x'))
    expected = pd.Series([1.0, 2.0, 3.0], index=[1, 2, 3])
    assert_series_equal(result, expected)
    result = results.predict({'x': [1, 2, 3]})
    expected = pd.Series([1.0, 2.0, 3.0], index=[0, 1, 2])
    assert_series_equal(result, expected, check_index_type=False)