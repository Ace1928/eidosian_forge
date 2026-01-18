import contextlib
import ctypes
import gc
import pyarrow as pa
import pytest
@needs_cffi
@pytest.mark.parametrize('reader_factory', [make_ipc_stream_reader, make_py_record_batch_reader])
def test_export_import_batch_reader(reader_factory):
    c_stream = ffi.new('struct ArrowArrayStream*')
    ptr_stream = int(ffi.cast('uintptr_t', c_stream))
    gc.collect()
    old_allocated = pa.total_allocated_bytes()
    _export_import_batch_reader(ptr_stream, reader_factory)
    assert pa.total_allocated_bytes() == old_allocated
    with assert_stream_released:
        pa.RecordBatchReader._import_from_c(ptr_stream)