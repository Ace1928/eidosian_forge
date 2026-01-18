from datetime import datetime
import re
import numpy as np
import pytest
from pandas.errors import IndexingError
import pandas.util._test_decorators as td
from pandas import (
import pandas._testing as tm
from pandas.api.types import is_scalar
from pandas.tests.indexing.common import check_indexing_smoketest_or_raises
def test_iloc_getitem_setitem_fancy_exceptions(self, float_frame):
    with pytest.raises(IndexingError, match='Too many indexers'):
        float_frame.iloc[:, :, :]
    with pytest.raises(IndexError, match='too many indices for array'):
        float_frame.iloc[:, :, :] = 1