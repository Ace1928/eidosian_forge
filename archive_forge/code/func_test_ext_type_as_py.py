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
def test_ext_type_as_py():
    ty = UuidType()
    expected = uuid4()
    scalar = pa.ExtensionScalar.from_storage(ty, expected.bytes)
    assert scalar.as_py() == expected
    uuids = [uuid4() for _ in range(3)]
    storage = pa.array([uuid.bytes for uuid in uuids], type=pa.binary(16))
    arr = pa.ExtensionArray.from_storage(ty, storage)
    for i, expected in enumerate(uuids):
        assert arr[i].as_py() == expected
    for result, expected in zip(arr, uuids):
        assert result.as_py() == expected
    data = [pa.ExtensionArray.from_storage(ty, storage), pa.ExtensionArray.from_storage(ty, storage)]
    carr = pa.chunked_array(data)
    for i, expected in enumerate(uuids + uuids):
        assert carr[i].as_py() == expected
    for result, expected in zip(carr, uuids + uuids):
        assert result.as_py() == expected