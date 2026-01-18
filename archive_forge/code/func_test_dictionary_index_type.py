import collections
import datetime
import decimal
import itertools
import math
import re
import sys
import hypothesis as h
import numpy as np
import pytest
from pyarrow.pandas_compat import _pandas_api  # noqa
import pyarrow as pa
from pyarrow.tests import util
import pyarrow.tests.strategies as past
@pytest.mark.parametrize('input_index_type', [pa.int8(), pa.int16(), pa.int32(), pa.int64()])
def test_dictionary_index_type(input_index_type):
    typ = pa.dictionary(input_index_type, value_type=pa.int64())
    arr = pa.array(range(10), type=typ)
    assert arr.type.equals(typ)