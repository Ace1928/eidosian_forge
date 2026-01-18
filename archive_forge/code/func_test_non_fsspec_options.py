import io
import numpy as np
import pytest
from pandas import (
import pandas._testing as tm
from pandas.util import _test_decorators as td
def test_non_fsspec_options():
    pytest.importorskip('pyarrow')
    with pytest.raises(ValueError, match='storage_options'):
        read_csv('localfile', storage_options={'a': True})
    with pytest.raises(ValueError, match='storage_options'):
        read_parquet('localfile', storage_options={'a': True})
    by = io.BytesIO()
    with pytest.raises(ValueError, match='storage_options'):
        read_csv(by, storage_options={'a': True})
    df = DataFrame({'a': [0]})
    with pytest.raises(ValueError, match='storage_options'):
        df.to_parquet('nonfsspecpath', storage_options={'a': True})