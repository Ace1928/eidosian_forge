from collections import OrderedDict
from collections.abc import Iterable
import sys
import weakref
import numpy as np
import pytest
import pyarrow as pa
import pyarrow.compute as pc
@pytest.mark.acero
def test_invalid_non_join_column():
    NUM_ITEMS = 30
    t1 = pa.Table.from_pydict({'id': range(NUM_ITEMS), 'array_column': [[z for z in range(3)] for x in range(NUM_ITEMS)]})
    t2 = pa.Table.from_pydict({'id': range(NUM_ITEMS), 'value': [x for x in range(NUM_ITEMS)]})
    with pytest.raises(pa.lib.ArrowInvalid) as excinfo:
        t1.join(t2, 'id', join_type='inner')
    exp_error_msg = 'Data type list<item: int64> is not supported ' + 'in join non-key field array_column'
    assert exp_error_msg in str(excinfo.value)
    with pytest.raises(pa.lib.ArrowInvalid) as excinfo:
        t2.join(t1, 'id', join_type='inner')
    assert exp_error_msg in str(excinfo.value)