from collections import OrderedDict
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_from_dict_orient_invalid(self):
    msg = "Expected 'index', 'columns' or 'tight' for orient parameter. Got 'abc' instead"
    with pytest.raises(ValueError, match=msg):
        DataFrame.from_dict({'foo': 1, 'baz': 3, 'bar': 2}, orient='abc')