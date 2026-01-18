import ctypes
import json
import os
import warnings
from typing import Any, Callable, Iterator, List, Optional, Sequence, Tuple, cast
import numpy as np
from ._typing import (
from .compat import DataFrame, lazy_isinstance
from .core import (
def record_batch_data_iter(data_iter: Iterator) -> Callable:
    """Data iterator used to ingest Arrow columnar record batches. We are not using
    class DataIter because it is only intended for building Device DMatrix and external
    memory DMatrix.

    """
    from pyarrow.cffi import ffi
    c_schemas: List[ffi.CData] = []
    c_arrays: List[ffi.CData] = []

    def _next(data_handle: int) -> int:
        from pyarrow.cffi import ffi
        try:
            batch = next(data_iter)
            c_schemas.append(ffi.new('struct ArrowSchema*'))
            c_arrays.append(ffi.new('struct ArrowArray*'))
            ptr_schema = int(ffi.cast('uintptr_t', c_schemas[-1]))
            ptr_array = int(ffi.cast('uintptr_t', c_arrays[-1]))
            batch._export_to_c(ptr_array, ptr_schema)
            _check_call(_LIB.XGImportArrowRecordBatch(ctypes.c_void_p(data_handle), ctypes.c_void_p(ptr_array), ctypes.c_void_p(ptr_schema)))
            return 1
        except StopIteration:
            return 0
    return _next