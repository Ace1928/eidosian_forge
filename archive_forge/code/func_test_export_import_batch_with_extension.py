import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
@needs_cffi
def test_export_import_batch_with_extension():
    with registered_extension_type(ParamExtType(1)):
        check_export_import_batch(make_extension_batch)