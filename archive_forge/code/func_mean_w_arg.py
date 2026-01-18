import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
def mean_w_arg(x, const):
    return np.mean(x) + const