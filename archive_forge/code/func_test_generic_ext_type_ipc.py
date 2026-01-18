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
def test_generic_ext_type_ipc(registered_period_type):
    period_type, period_class = registered_period_type
    storage = pa.array([1, 2, 3, 4], pa.int64())
    arr = pa.ExtensionArray.from_storage(period_type, storage)
    batch = pa.RecordBatch.from_arrays([arr], ['ext'])
    assert isinstance(arr, period_class)
    buf = ipc_write_batch(batch)
    del batch
    batch = ipc_read_batch(buf)
    result = batch.column(0)
    assert isinstance(result, period_class)
    assert result.type.extension_name == 'test.period'
    assert arr.storage.to_pylist() == [1, 2, 3, 4]
    assert isinstance(result.type, PeriodType)
    assert result.type.freq == 'D'
    assert result.type == period_type
    period_type_H = period_type.__class__('H')
    assert period_type_H.extension_name == 'test.period'
    assert period_type_H.freq == 'H'
    arr = pa.ExtensionArray.from_storage(period_type_H, storage)
    batch = pa.RecordBatch.from_arrays([arr], ['ext'])
    buf = ipc_write_batch(batch)
    del batch
    batch = ipc_read_batch(buf)
    result = batch.column(0)
    assert isinstance(result.type, PeriodType)
    assert result.type.freq == 'H'
    assert isinstance(result, period_class)