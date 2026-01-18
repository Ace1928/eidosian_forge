import inspect
import operator
import numpy as np
import pytest
from pandas._typing import Dtype
from pandas.core.dtypes.common import is_bool_dtype
from pandas.core.dtypes.dtypes import NumpyEADtype
from pandas.core.dtypes.missing import na_value_for_dtype
import pandas as pd
import pandas._testing as tm
from pandas.core.sorting import nargsort
@pytest.mark.parametrize('method', ['argmax', 'argmin'])
def test_argmin_argmax_empty_array(self, method, data):
    err_msg = 'attempt to get'
    with pytest.raises(ValueError, match=err_msg):
        getattr(data[:0], method)()