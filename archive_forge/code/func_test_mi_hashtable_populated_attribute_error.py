import re
import numpy as np
import pytest
from pandas._libs import index as libindex
from pandas.core.dtypes.cast import construct_1d_object_array_from_listlike
import pandas as pd
from pandas import (
import pandas._testing as tm
def test_mi_hashtable_populated_attribute_error(monkeypatch):
    monkeypatch.setattr(libindex, '_SIZE_CUTOFF', 50)
    r = range(50)
    df = pd.DataFrame({'a': r, 'b': r}, index=MultiIndex.from_arrays([r, r]))
    msg = "'Series' object has no attribute 'foo'"
    with pytest.raises(AttributeError, match=msg):
        df['a'].foo()