import numpy as np
import pytest
from pandas.core.dtypes.dtypes import CategoricalDtype
from pandas import (
import pandas._testing as tm
def test_iter_python_types_datetime(self):
    cat = Categorical([Timestamp('2017-01-01'), Timestamp('2017-01-02')])
    assert isinstance(next(iter(cat)), Timestamp)
    assert isinstance(cat.tolist()[0], Timestamp)