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
def test_ext_array_errors():
    ty = ParamExtType(4)
    storage = pa.array([b'foo', b'bar'], type=pa.binary(3))
    with pytest.raises(TypeError, match='Incompatible storage type'):
        pa.ExtensionArray.from_storage(ty, storage)