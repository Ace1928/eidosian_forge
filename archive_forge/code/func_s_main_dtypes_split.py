from itertools import product
import numpy as np
import pytest
import pandas as pd
from pandas import Series
import pandas._testing as tm
@pytest.fixture(params=main_dtypes)
def s_main_dtypes_split(request, s_main_dtypes):
    """Each series in s_main_dtypes."""
    return s_main_dtypes[request.param]