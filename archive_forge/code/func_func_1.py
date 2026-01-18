import numpy as np
import pytest
from pandas.errors import NumbaUtilError
from pandas import (
import pandas._testing as tm
def func_1(values, index):
    return values + 1