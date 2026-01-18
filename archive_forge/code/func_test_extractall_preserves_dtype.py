from datetime import datetime
import re
import numpy as np
import pytest
from pandas.core.dtypes.dtypes import ArrowDtype
from pandas import (
def test_extractall_preserves_dtype():
    pa = pytest.importorskip('pyarrow')
    result = Series(['abc', 'ab'], dtype=ArrowDtype(pa.string())).str.extractall('(ab)')
    assert result.dtypes[0] == 'string[pyarrow]'