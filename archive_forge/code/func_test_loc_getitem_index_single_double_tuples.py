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
@pytest.mark.parametrize('tpl', [(1,), (1, 2)])
def test_loc_getitem_index_single_double_tuples(self, tpl):
    idx = Index([(1,), (1, 2)], name='A', tupleize_cols=False)
    df = DataFrame(index=idx)
    result = df.loc[[tpl]]
    idx = Index([tpl], name='A', tupleize_cols=False)
    expected = DataFrame(index=idx)
    tm.assert_frame_equal(result, expected)