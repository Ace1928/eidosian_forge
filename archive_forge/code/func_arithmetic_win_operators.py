from datetime import (
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
@pytest.fixture(params=['sum', 'mean', 'median', 'max', 'min', 'var', 'std', 'kurt', 'skew', 'count', 'sem'])
def arithmetic_win_operators(request):
    return request.param