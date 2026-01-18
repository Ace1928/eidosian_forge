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
@pytest.mark.parametrize('cast_as_obj', [True, False])
@pytest.mark.parametrize('index', [date_range('2015-01-01 10:00', freq='D', periods=3, tz='US/Eastern', name='Green Eggs & Ham'), date_range('2015-01-01 10:00', freq='D', periods=3), timedelta_range('1 days', freq='D', periods=3), period_range('2015-01-01', freq='D', periods=3)])
def test_constructor_from_index_dtlike(self, cast_as_obj, index):
    if cast_as_obj:
        with tm.assert_produces_warning(FutureWarning, match='Dtype inference'):
            result = Index(index.astype(object))
    else:
        result = Index(index)
    tm.assert_index_equal(result, index)
    if isinstance(index, DatetimeIndex):
        assert result.tz == index.tz
        if cast_as_obj:
            index += pd.Timedelta(nanoseconds=50)
            result = Index(index, dtype=object)
            assert result.dtype == np.object_
            assert list(result) == list(index)