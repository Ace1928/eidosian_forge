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
def test_convert_categories_to_array_with_string_pyarrow_dtype():
    if Version(pd.__version__) < Version('1.3.0'):
        pytest.skip('PyArrow backed string data type introduced in pandas 1.3.0')
    df = pd.DataFrame({'x': ['foo', 'bar', 'foo']}, dtype='string[pyarrow]')
    df = df.astype('category')
    indices = pa.array(df['x'].cat.codes)
    dictionary = pa.array(df['x'].cat.categories.values)
    assert isinstance(dictionary, pa.Array)
    expected = pa.Array.from_pandas(df['x'])
    result = pa.DictionaryArray.from_arrays(indices, dictionary)
    assert result == expected