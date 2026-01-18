import numpy as np
import pytest
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.util.hashing import hash_tuples
from pandas.util import (
def test_hash_array_index_exception():
    obj = pd.DatetimeIndex(['2018-10-28 01:20:00'], tz='Europe/Berlin')
    msg = 'Use hash_pandas_object instead'
    with pytest.raises(TypeError, match=msg):
        hash_array(obj)