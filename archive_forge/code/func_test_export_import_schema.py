import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
@needs_cffi
def test_export_import_schema():
    check_export_import_schema(make_schema)