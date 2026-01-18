from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_chunked_array_mismatch_types():
    msg = 'chunks must all be same type'
    with pytest.raises(TypeError, match=msg):
        pa.chunked_array([pa.array([1, 2, 3]), pa.array([1.0, 2.0, 3.0])])
    with pytest.raises(TypeError, match=msg):
        pa.chunked_array([pa.array([1, 2, 3])], type=pa.float64())