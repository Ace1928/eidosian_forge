from functools import partial
import numpy as np
import pytest
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import is_integer_dtype
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core import nanops
@pytest.mark.parametrize('fixture', ['arr_float', 'arr_complex', 'arr_int', 'arr_bool', 'arr_str', 'arr_utf'])
def test_bn_ok_dtype(fixture, request, disable_bottleneck):
    obj = request.getfixturevalue(fixture)
    assert nanops._bn_ok_dtype(obj.dtype, 'test')