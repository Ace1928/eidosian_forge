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
def test_buffer_bounds_error(version):
    path = random_path()
    TEST_FILES.append(path)
    for i in range(16, 256):
        table = pa.Table.from_arrays([pa.array([None] + list(range(i)), type=pa.float64())], names=['arr'])
        _check_arrow_roundtrip(table)