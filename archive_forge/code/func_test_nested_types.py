import io
import os
import sys
import tempfile
import pytest
import hypothesis as h
import hypothesis.strategies as st
import numpy as np
import pyarrow as pa
import pyarrow.tests.strategies as past
from pyarrow.feather import (read_feather, write_feather, read_table,
def test_nested_types(compression):
    table = pa.table({'col': pa.StructArray.from_arrays([[0, 1, 2], [1, 2, 3]], names=['f1', 'f2'])})
    _check_arrow_roundtrip(table, compression=compression)
    table = pa.table({'col': pa.array([[1, 2], [3, 4]])})
    _check_arrow_roundtrip(table, compression=compression)
    table = pa.table({'col': pa.array([[[1, 2], [3, 4]], [[5, 6], None]])})
    _check_arrow_roundtrip(table, compression=compression)