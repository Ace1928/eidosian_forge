from collections import defaultdict
from datetime import datetime
from functools import partial
import math
import operator
import re
import numpy as np
import pytest
from pandas.compat import IS64
from pandas.errors import InvalidIndexError
import pandas.util._test_decorators as td
from pandas.core.dtypes.common import (
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.core.indexes.api import (
@pytest.mark.parametrize('index', ['string', 'int64', 'int32', 'uint64', 'uint32', 'float64', 'float32'], indirect=True)
@pytest.mark.parametrize('dtype', [int, np.bool_])
def test_empty_fancy(self, index, dtype, request, using_infer_string):
    if dtype is np.bool_ and using_infer_string and (index.dtype == 'string'):
        request.applymarker(pytest.mark.xfail(reason='numpy behavior is buggy'))
    empty_arr = np.array([], dtype=dtype)
    empty_index = type(index)([], dtype=index.dtype)
    assert index[[]].identical(empty_index)
    if dtype == np.bool_:
        with tm.assert_produces_warning(FutureWarning, match='is deprecated'):
            assert index[empty_arr].identical(empty_index)
    else:
        assert index[empty_arr].identical(empty_index)