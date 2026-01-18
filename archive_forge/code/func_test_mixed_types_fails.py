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
@pytest.mark.parametrize('data,error_type', [({'a': ['a', 1, 2.0]}, pa.ArrowTypeError), ({'a': ['a', 1, 2.0]}, pa.ArrowTypeError), ({'a': [1, True]}, pa.ArrowTypeError), ({'a': [True, 'a']}, pa.ArrowInvalid), ({'a': [1, 'a']}, pa.ArrowInvalid), ({'a': [1.0, 'a']}, pa.ArrowInvalid)])
def test_mixed_types_fails(self, data, error_type):
    df = pd.DataFrame(data)
    msg = 'Conversion failed for column a with type object'
    with pytest.raises(error_type, match=msg):
        pa.Table.from_pandas(df)