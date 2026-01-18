from datetime import timedelta
import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.errors import (
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_contains_with_nat(self):
    mi = MultiIndex(levels=[['C'], date_range('2012-01-01', periods=5)], codes=[[0, 0, 0, 0, 0, 0], [-1, 0, 1, 2, 3, 4]], names=[None, 'B'])
    assert ('C', pd.Timestamp('2012-01-01')) in mi
    for val in mi.values:
        assert val in mi