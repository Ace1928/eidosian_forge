from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
def test_invalid_table_construct():
    array = np.array([0, 1], dtype=np.uint8)
    u8 = pa.uint8()
    arrays = [pa.array(array, type=u8), pa.array(array[1:], type=u8)]
    with pytest.raises(pa.lib.ArrowInvalid):
        pa.Table.from_arrays(arrays, names=['a1', 'a2'])