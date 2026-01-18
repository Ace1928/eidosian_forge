from datetime import datetime
import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.algorithms import safe_sort
def test_intersection_str_dates(self, sort):
    dt_dates = [datetime(2012, 2, 9), datetime(2012, 2, 22)]
    i1 = Index(dt_dates, dtype=object)
    i2 = Index(['aa'], dtype=object)
    result = i2.intersection(i1, sort=sort)
    assert len(result) == 0