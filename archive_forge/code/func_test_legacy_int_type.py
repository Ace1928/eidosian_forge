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
def test_legacy_int_type():
    with pytest.warns(FutureWarning, match='PyExtensionType is deprecated'):
        ext_ty = LegacyIntType()
    arr = pa.array([1, 2, 3], type=ext_ty.storage_type)
    ext_arr = pa.ExtensionArray.from_storage(ext_ty, arr)
    batch = pa.RecordBatch.from_arrays([ext_arr], names=['ext'])
    buf = ipc_write_batch(batch)
    with pytest.warns((RuntimeWarning, FutureWarning)):
        batch = ipc_read_batch(buf)
        assert isinstance(batch.column(0).type, pa.UnknownExtensionType)
    with enabled_auto_load():
        with pytest.warns(FutureWarning, match='PyExtensionType is deprecated'):
            batch = ipc_read_batch(buf)
            assert isinstance(batch.column(0).type, LegacyIntType)
            assert batch.column(0) == ext_arr