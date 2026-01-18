import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
@needs_cffi
def test_export_import_field():
    c_schema = ffi.new('struct ArrowSchema*')
    ptr_schema = int(ffi.cast('uintptr_t', c_schema))
    gc.collect()
    old_allocated = pa.total_allocated_bytes()
    field = pa.field('test', pa.list_(pa.int32()), nullable=True)
    field._export_to_c(ptr_schema)
    assert pa.total_allocated_bytes() > old_allocated
    del field
    assert pa.total_allocated_bytes() > old_allocated
    field_new = pa.Field._import_from_c(ptr_schema)
    assert field_new == pa.field('test', pa.list_(pa.int32()), nullable=True)
    assert pa.total_allocated_bytes() == old_allocated
    with assert_schema_released:
        pa.Field._import_from_c(ptr_schema)