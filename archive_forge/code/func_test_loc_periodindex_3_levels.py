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
@pytest.mark.filterwarnings('ignore:Period with BDay freq is deprecated:FutureWarning')
@pytest.mark.filterwarnings('ignore:PeriodDtype\\[B\\] is deprecated:FutureWarning')
def test_loc_periodindex_3_levels():
    p_index = PeriodIndex(['20181101 1100', '20181101 1200', '20181102 1300', '20181102 1400'], name='datetime', freq='B')
    mi_series = DataFrame([['A', 'B', 1.0], ['A', 'C', 2.0], ['Z', 'Q', 3.0], ['W', 'F', 4.0]], index=p_index, columns=['ONE', 'TWO', 'VALUES'])
    mi_series = mi_series.set_index(['ONE', 'TWO'], append=True)['VALUES']
    assert mi_series.loc[p_index[0], 'A', 'B'] == 1.0