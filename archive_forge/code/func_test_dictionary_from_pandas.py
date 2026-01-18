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
def test_dictionary_from_pandas():
    cat = pd.Categorical(['a', 'b', 'a'])
    expected_type = pa.dictionary(pa.int8(), pa.string())
    result = pa.array(cat)
    assert result.to_pylist() == ['a', 'b', 'a']
    assert result.type.equals(expected_type)
    cat = pd.Categorical(['a', 'b', None, 'a'])
    result = pa.array(cat)
    assert result.to_pylist() == ['a', 'b', None, 'a']
    assert result.type.equals(expected_type)
    result = pa.array(cat, mask=np.array([False, False, False, True]))
    assert result.to_pylist() == ['a', 'b', None, None]
    assert result.type.equals(expected_type)