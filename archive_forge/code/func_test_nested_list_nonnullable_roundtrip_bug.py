import decimal
import io
import numpy as np
import pytest
import pyarrow as pa
from pyarrow.tests import util
from pyarrow.tests.parquet.common import _check_roundtrip
def test_nested_list_nonnullable_roundtrip_bug():
    typ = pa.list_(pa.field('item', pa.float32(), False))
    num_rows = 10000
    t = pa.table([pa.array([[0] * ((i + 5) % 10) for i in range(0, 10)] * (num_rows // 10), type=typ)], ['a'])
    _check_roundtrip(t, data_page_size=4096)