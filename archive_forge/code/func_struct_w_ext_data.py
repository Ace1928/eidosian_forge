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
@pytest.fixture
def struct_w_ext_data():
    storage1 = pa.array([1, 2, 3], type=pa.int64())
    storage2 = pa.array([b'123', b'456', b'789'], type=pa.binary(3))
    ty1 = IntegerType()
    ty2 = ParamExtType(3)
    arr1 = pa.ExtensionArray.from_storage(ty1, storage1)
    arr2 = pa.ExtensionArray.from_storage(ty2, storage2)
    sarr1 = pa.StructArray.from_arrays([arr1], ['f0'])
    sarr2 = pa.StructArray.from_arrays([arr2], ['f1'])
    return [sarr1, sarr2]