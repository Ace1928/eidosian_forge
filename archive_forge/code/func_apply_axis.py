import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
@pytest.fixture(params=[0, 1])
def apply_axis(request):
    return request.param