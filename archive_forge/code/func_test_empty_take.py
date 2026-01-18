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
def test_empty_take():
    ext_type = IntegerType()
    storage = pa.array([], type=pa.int64())
    empty_arr = pa.ExtensionArray.from_storage(ext_type, storage)
    result = empty_arr.filter(pa.array([], pa.bool_()))
    assert len(result) == 0
    assert result.equals(empty_arr)
    result = empty_arr.take(pa.array([], pa.int32()))
    assert len(result) == 0
    assert result.equals(empty_arr)