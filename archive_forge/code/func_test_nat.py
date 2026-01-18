import datetime as dt
from datetime import date
import re
import numpy as np
import pytest
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_nat(self):
    assert DatetimeIndex([np.nan])[0] is pd.NaT