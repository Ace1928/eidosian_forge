from collections import namedtuple
from datetime import (
import re
from dateutil.tz import gettz
import numpy as np
import pytest
from pandas._config import using_pyarrow_string_dtype
from pandas._libs import index as libindex
from pandas.compat.numpy import np_version_gt2
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
import pandas as pd
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.core.indexing import _one_ellipsis_message
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
@pytest.mark.parametrize('obj, key, exp', [(DataFrame([[1]], columns=Index([False])), IndexSlice[:, False], Series([1], name=False)), (Series([1], index=Index([False])), False, [1]), (DataFrame([[1]], index=Index([False])), False, Series([1], name=False))])
def test_loc_getitem_single_boolean_arg(self, obj, key, exp):
    res = obj.loc[key]
    if isinstance(exp, (DataFrame, Series)):
        tm.assert_equal(res, exp)
    else:
        assert res == exp