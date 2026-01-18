import datetime as dt
from datetime import date
import re
import numpy as np
import pytest
from pandas.compat.numpy import np_long
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_misc_coverage(self):
    rng = date_range('1/1/2000', periods=5)
    result = rng.groupby(rng.day)
    assert isinstance(next(iter(result.values()))[0], Timestamp)