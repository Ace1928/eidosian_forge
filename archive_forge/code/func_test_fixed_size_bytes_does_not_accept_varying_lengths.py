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
def test_fixed_size_bytes_does_not_accept_varying_lengths():
    data = [b'foo', None, b'barb', b'2346']
    with pytest.raises(pa.ArrowInvalid):
        pa.array(data, type=pa.binary(4))