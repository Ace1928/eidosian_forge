from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.validation import (
from statsmodels.tools.validation.validation import _right_squeeze
@pytest.fixture(params=(3.2, np.float32(3.2), 3 + 2j, complex(2.3 + 0j), 'apple', 1.0 + 0j, np.timedelta64(2)))
def not_integer(request):
    return request.param