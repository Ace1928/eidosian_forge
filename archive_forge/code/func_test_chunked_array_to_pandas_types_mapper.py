import gc
import decimal
import json
import multiprocessing as mp
import sys
import warnings
from collections import OrderedDict
from datetime import date, datetime, time, timedelta, timezone
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import numpy.testing as npt
import pytest
from pyarrow.pandas_compat import get_logical_type, _pandas_api
from pyarrow.tests.util import invoke_script, random_ascii, rands
import pyarrow.tests.strategies as past
import pyarrow.tests.util as test_util
from pyarrow.vendored.version import Version
import pyarrow as pa
@pytest.mark.pandas
def test_chunked_array_to_pandas_types_mapper():
    if Version(pd.__version__) < Version('1.2.0'):
        pytest.skip('Float64Dtype extension dtype missing')
    data = pa.chunked_array([pa.array([1, 2, 3], pa.int64())])
    assert isinstance(data, pa.ChunkedArray)
    types_mapper = {pa.int64(): pd.Int64Dtype()}.get
    result = data.to_pandas(types_mapper=types_mapper)
    assert result.dtype == pd.Int64Dtype()
    types_mapper = {pa.int64(): None}.get
    result = data.to_pandas(types_mapper=types_mapper)
    assert result.dtype == np.dtype('int64')
    types_mapper = {pa.float64(): pd.Float64Dtype()}.get
    result = data.to_pandas(types_mapper=types_mapper)
    assert result.dtype == np.dtype('int64')