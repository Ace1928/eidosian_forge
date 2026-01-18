import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import CategoricalDtype
import pandas.core.reshape.tile as tmod
def test_datetime_nan_error():
    msg = 'bins must be of datetime64 dtype'
    with pytest.raises(ValueError, match=msg):
        cut(date_range('20130101', periods=3), bins=[0, 2, 4])