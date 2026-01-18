from io import StringIO
from dateutil.parser import parse
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
def convert_score(x):
    x = x.strip()
    if not x:
        return np.nan
    if x.find('-') > 0:
        val_min, val_max = map(int, x.split('-'))
        val = 0.5 * (val_min + val_max)
    else:
        val = float(x)
    return val