import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
def make_extension_schema():
    return pa.schema([('ext', ParamExtType(3))], metadata={b'key1': b'value1'})