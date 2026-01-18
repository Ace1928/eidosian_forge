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
@pytest.mark.parametrize('lt_value', [30, 10])
def test_loc_multiindex_levels_contain_values_not_in_index_anymore(self, lt_value):
    df = DataFrame({'a': [12, 23, 34, 45]}, index=[list('aabb'), [0, 1, 2, 3]])
    with pytest.raises(KeyError, match="\\['b'\\] not in index"):
        df.loc[df['a'] < lt_value, :].loc[['b'], :]