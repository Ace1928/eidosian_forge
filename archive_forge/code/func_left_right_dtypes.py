import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.arrays import IntervalArray
@pytest.fixture(params=[(Index([0, 2, 4]), Index([1, 3, 5])), (Index([0.0, 1.0, 2.0]), Index([1.0, 2.0, 3.0])), (timedelta_range('0 days', periods=3), timedelta_range('1 day', periods=3)), (date_range('20170101', periods=3), date_range('20170102', periods=3)), (date_range('20170101', periods=3, tz='US/Eastern'), date_range('20170102', periods=3, tz='US/Eastern'))], ids=lambda x: str(x[0].dtype))
def left_right_dtypes(request):
    """
    Fixture for building an IntervalArray from various dtypes
    """
    return request.param