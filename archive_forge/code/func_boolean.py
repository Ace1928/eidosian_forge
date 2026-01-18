from collections import OrderedDict
import numpy as np
import pandas as pd
import pytest
from statsmodels.tools.validation import (
from statsmodels.tools.validation.validation import _right_squeeze
@pytest.fixture(params=[True, False, 1, 1.2, 'a', ''])
def boolean(request):
    return request.param