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
def test_is_scalar_access(self):
    index = Index([1, 2, 1])
    ser = Series(range(3), index=index)
    assert ser.iloc._is_scalar_access((1,))
    df = ser.to_frame()
    assert df.iloc._is_scalar_access((1, 0))