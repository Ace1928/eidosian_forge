import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
@needs_cffi
def test_export_import_type():
    c_schema = ffi.new('struct ArrowSchema*')
    ptr_schema = int(ffi.cast('uintptr_t', c_schema))
    gc.collect()
    old_allocated = pa.total_allocated_bytes()
    typ = pa.list_(pa.int32())
    typ._export_to_c(ptr_schema)
    assert pa.total_allocated_bytes() > old_allocated
    del typ
    assert pa.total_allocated_bytes() > old_allocated
    typ_new = pa.DataType._import_from_c(ptr_schema)
    assert typ_new == pa.list_(pa.int32())
    assert pa.total_allocated_bytes() == old_allocated
    with assert_schema_released:
        pa.DataType._import_from_c(ptr_schema)
    pa.int32()._export_to_c(ptr_schema)
    bad_format = ffi.new('char[]', b'zzz')
    c_schema.format = bad_format
    with pytest.raises(ValueError, match='Invalid or unsupported format string'):
        pa.DataType._import_from_c(ptr_schema)
    with assert_schema_released:
        pa.DataType._import_from_c(ptr_schema)