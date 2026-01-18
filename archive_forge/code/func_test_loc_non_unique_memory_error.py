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
@pytest.mark.arm_slow
@pytest.mark.parametrize('length, l2', [[900, 100], [900000, 100000]])
def test_loc_non_unique_memory_error(self, length, l2):
    columns = list('ABCDEFG')
    df = pd.concat([DataFrame(np.random.default_rng(2).standard_normal((length, len(columns))), index=np.arange(length), columns=columns), DataFrame(np.ones((l2, len(columns))), index=[0] * l2, columns=columns)])
    assert df.index.is_unique is False
    mask = np.arange(l2)
    result = df.loc[mask]
    expected = pd.concat([df.take([0]), DataFrame(np.ones((len(mask), len(columns))), index=[0] * len(mask), columns=columns), df.take(mask[1:])])
    tm.assert_frame_equal(result, expected)