import numpy as np
import pytest
from pandas.errors import NumbaUtilError
from pandas import (
import pandas._testing as tm
def func_numba(values, index):
    return np.mean(values) * 2.7