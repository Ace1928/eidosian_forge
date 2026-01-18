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
def test_loc_setitem_datetime_coercion(self):
    df = DataFrame({'c': [Timestamp('2010-10-01')] * 3})
    df.loc[0:1, 'c'] = np.datetime64('2008-08-08')
    assert Timestamp('2008-08-08') == df.loc[0, 'c']
    assert Timestamp('2008-08-08') == df.loc[1, 'c']
    with tm.assert_produces_warning(FutureWarning, match='incompatible dtype'):
        df.loc[2, 'c'] = date(2005, 5, 5)
    assert Timestamp('2005-05-05').date() == df.loc[2, 'c']