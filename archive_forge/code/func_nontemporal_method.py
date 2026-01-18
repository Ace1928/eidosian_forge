import numpy as np
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture(params=['linear', 'index', 'values', 'nearest', 'slinear', 'zero', 'quadratic', 'cubic', 'barycentric', 'krogh', 'polynomial', 'spline', 'piecewise_polynomial', 'from_derivatives', 'pchip', 'akima', 'cubicspline'])
def nontemporal_method(request):
    """Fixture that returns an (method name, required kwargs) pair.

    This fixture does not include method 'time' as a parameterization; that
    method requires a Series with a DatetimeIndex, and is generally tested
    separately from these non-temporal methods.
    """
    method = request.param
    kwargs = {'order': 1} if method in ('spline', 'polynomial') else {}
    return (method, kwargs)