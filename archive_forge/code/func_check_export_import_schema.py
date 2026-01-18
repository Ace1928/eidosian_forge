import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
def check_export_import_schema(schema_factory, expected_schema_factory=None):
    if expected_schema_factory is None:
        expected_schema_factory = schema_factory
    c_schema = ffi.new('struct ArrowSchema*')
    ptr_schema = int(ffi.cast('uintptr_t', c_schema))
    gc.collect()
    old_allocated = pa.total_allocated_bytes()
    schema_factory()._export_to_c(ptr_schema)
    assert pa.total_allocated_bytes() > old_allocated
    schema_new = pa.Schema._import_from_c(ptr_schema)
    assert schema_new == expected_schema_factory()
    assert pa.total_allocated_bytes() == old_allocated
    del schema_new
    assert pa.total_allocated_bytes() == old_allocated
    with assert_schema_released:
        pa.Schema._import_from_c(ptr_schema)
    pa.int32()._export_to_c(ptr_schema)
    with pytest.raises(ValueError, match='ArrowSchema describes non-struct type'):
        pa.Schema._import_from_c(ptr_schema)
    with assert_schema_released:
        pa.Schema._import_from_c(ptr_schema)