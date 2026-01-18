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
@pytest.mark.parametrize('data,ty', (([1, 2], pa.int32), ([1, 2], pa.int64), (['1', '2'], pa.string), ([b'1', b'2'], pa.binary), ([1.0, 2.0], pa.float32), ([1.0, 2.0], pa.float64)))
def test_casting_to_extension_type(data, ty):
    arr = pa.array(data, ty())
    out = arr.cast(IntegerType())
    assert isinstance(out, pa.ExtensionArray)
    assert out.type == IntegerType()
    assert out.to_pylist() == [1, 2]