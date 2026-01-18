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
def test_ext_array_to_pylist():
    ty = ParamExtType(3)
    storage = pa.array([b'foo', b'bar', None], type=pa.binary(3))
    arr = pa.ExtensionArray.from_storage(ty, storage)
    assert arr.to_pylist() == [b'foo', b'bar', None]