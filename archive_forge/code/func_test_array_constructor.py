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
def test_array_constructor():
    ext_type = IntegerType()
    storage = pa.array([1, 2, 3], type=pa.int64())
    expected = pa.ExtensionArray.from_storage(ext_type, storage)
    result = pa.array([1, 2, 3], type=IntegerType())
    assert result.equals(expected)
    result = pa.array(np.array([1, 2, 3]), type=IntegerType())
    assert result.equals(expected)
    result = pa.array(np.array([1.0, 2.0, 3.0]), type=IntegerType())
    assert result.equals(expected)