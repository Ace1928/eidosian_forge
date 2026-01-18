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
def check_example_batch(batch, *, expect_extension):
    arr = batch.column(0)
    if expect_extension:
        assert isinstance(arr, pa.ExtensionArray)
        assert arr.type.storage_type == pa.binary(3)
        assert arr.storage.to_pylist() == [b'foo', b'bar']
    else:
        assert arr.type == pa.binary(3)
        assert arr.to_pylist() == [b'foo', b'bar']
    return arr