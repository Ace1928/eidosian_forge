from io import StringIO
from dateutil.parser import parse
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def convert_days_sentinel(x):
    x = x.strip()
    if not x:
        return np.nan
    is_plus = x.endswith('+')
    if is_plus:
        x = int(x[:-1]) + 1
    else:
        x = int(x)
    return x