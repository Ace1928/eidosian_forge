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
def test_loc_getitem_slice_floats_inexact(self):
    index = [52195.504153, 52196.303147, 52198.369883]
    df = DataFrame(np.random.default_rng(2).random((3, 2)), index=index)
    s1 = df.loc[52195.1:52196.5]
    assert len(s1) == 2
    s1 = df.loc[52195.1:52196.6]
    assert len(s1) == 2
    s1 = df.loc[52195.1:52198.9]
    assert len(s1) == 3