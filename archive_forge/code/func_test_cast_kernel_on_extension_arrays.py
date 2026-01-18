import contextlib
import os
import shutil
import subprocess
import weakref
from uuid import uuid4, UUID
import sys
import numpy as np
import pyarrow as pa
from pyarrow.vendored.version import Version
import pytest
def test_cast_kernel_on_extension_arrays():
    storage = pa.array([1, 2, 3, 4], pa.int64())
    arr = pa.ExtensionArray.from_storage(IntegerType(), storage)
    allocated_before_cast = pa.total_allocated_bytes()
    casted = arr.cast(pa.int64())
    assert pa.total_allocated_bytes() == allocated_before_cast
    cases = [(pa.int64(), pa.Int64Array), (pa.int32(), pa.Int32Array), (pa.int16(), pa.Int16Array), (pa.uint64(), pa.UInt64Array), (pa.uint32(), pa.UInt32Array), (pa.uint16(), pa.UInt16Array)]
    for typ, klass in cases:
        casted = arr.cast(typ)
        assert casted.type == typ
        assert isinstance(casted, klass)
    arr = pa.chunked_array([arr, arr])
    casted = arr.cast(pa.int16())
    assert casted.type == pa.int16()
    assert isinstance(casted, pa.ChunkedArray)