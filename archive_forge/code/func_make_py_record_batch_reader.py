import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
def make_py_record_batch_reader(schema, batches):
    return pa.RecordBatchReader.from_batches(schema, batches)