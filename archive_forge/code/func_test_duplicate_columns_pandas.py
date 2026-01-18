import io
import os
import sys
import tempfile
import pytest
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
from pyarrow.feather import (read_feather, write_feather, read_table,
@pytest.mark.pandas
def test_duplicate_columns_pandas():
    df = pd.DataFrame(np.arange(12).reshape(4, 3), columns=list('aaa')).copy()
    _assert_error_on_write(df, ValueError)