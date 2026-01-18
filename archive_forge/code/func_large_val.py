import decimal
import numpy as np
from numpy import iinfo
import pytest
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
@pytest.fixture(params=[47393996303418497800, 100000000000000000000])
def large_val(request):
    return request.param