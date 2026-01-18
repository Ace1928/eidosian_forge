import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
def make_ipc_stream_reader(schema, batches):
    return pa.ipc.open_stream(make_serialized(schema, batches))