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
def test_delete_partial_file_on_error(version):
    if sys.platform == 'win32':
        pytest.skip('Windows hangs on to file handle for some reason')

    class CustomClass:
        pass
    df = pd.DataFrame({'numbers': range(5), 'strings': [b'foo', None, 'bar', CustomClass(), np.nan]}, columns=['numbers', 'strings'])
    path = random_path()
    try:
        write_feather(df, path, version=version)
    except Exception:
        pass
    assert not os.path.exists(path)