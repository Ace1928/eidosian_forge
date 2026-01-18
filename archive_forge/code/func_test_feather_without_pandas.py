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
def test_feather_without_pandas(tempdir, version):
    table = pa.table([pa.array([1, 2, 3])], names=['f0'])
    path = str(tempdir / 'data.feather')
    _check_arrow_roundtrip(table, path)