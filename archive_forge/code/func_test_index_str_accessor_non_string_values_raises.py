from datetime import (
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.core.strings.accessor import StringMethods
from pandas.tests.strings import object_pyarrow_numpy
@pytest.mark.parametrize('values,inferred_type', [([1, np.nan], 'floating'), ([datetime(2011, 1, 1)], 'datetime64'), ([timedelta(1)], 'timedelta64')])
def test_index_str_accessor_non_string_values_raises(values, inferred_type, index_or_series):
    obj = index_or_series(values)
    if index_or_series is Index:
        assert obj.inferred_type == inferred_type
    msg = 'Can only use .str accessor with string values'
    with pytest.raises(AttributeError, match=msg):
        obj.str