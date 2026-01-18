from collections.abc import Iterable
import datetime
import decimal
import hypothesis as h
import hypothesis.strategies as st
import itertools
import pytest
import struct
import subprocess
import sys
import weakref
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
@pytest.mark.pandas
def test_array_supported_pandas_masks():
    import pandas
    arr = pa.array(pandas.Series([0, 1], name='a', dtype='int64'), mask=pandas.Series([True, False], dtype='bool'))
    assert arr.to_pylist() == [None, 1]