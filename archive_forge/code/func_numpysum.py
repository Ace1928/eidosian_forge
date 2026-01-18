import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.tseries import offsets
def numpysum(x, par):
    return np.sum(x + par)