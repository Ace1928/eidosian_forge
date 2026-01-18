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
@pytest.mark.parametrize('indexer', [IndexSlice['A', :], ('A', slice(None))])
def test_loc_series_getitem_too_many_dimensions(self, indexer):
    ser = Series(index=MultiIndex.from_tuples([('A', '0'), ('A', '1'), ('B', '0')]), data=[21, 22, 23])
    msg = 'Too many indexers'
    with pytest.raises(IndexingError, match=msg):
        ser.loc[indexer, :]
    with pytest.raises(IndexingError, match=msg):
        ser.loc[indexer, :] = 1