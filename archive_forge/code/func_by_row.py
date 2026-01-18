import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.tests.apply.common import series_transform_kernels
@pytest.fixture(params=[False, 'compat'])
def by_row(request):
    return request.param