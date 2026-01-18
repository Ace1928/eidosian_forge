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
def test_ext_array_lifetime():
    ty = ParamExtType(3)
    storage = pa.array([b'foo', b'bar'], type=pa.binary(3))
    arr = pa.ExtensionArray.from_storage(ty, storage)
    refs = [weakref.ref(ty), weakref.ref(arr), weakref.ref(storage)]
    del ty, storage, arr
    for ref in refs:
        assert ref() is None