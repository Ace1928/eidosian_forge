import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
def make_schema():
    return pa.schema([('ints', pa.list_(pa.int32()))], metadata={b'key1': b'value1'})