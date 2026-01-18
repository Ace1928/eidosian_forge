from collections import OrderedDict
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas import (
import pandas._testing as tm
def test_from_dict_scalars_requires_index(self):
    msg = 'If using all scalar values, you must pass an index'
    with pytest.raises(ValueError, match=msg):
        DataFrame.from_dict(OrderedDict([('b', 8), ('a', 5), ('a', 6)]))