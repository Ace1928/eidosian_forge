from string import ascii_lowercase
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def raise_if_sum_is_zero(x):
    if x.sum() == 0:
        raise ValueError
    return x.sum() > 0