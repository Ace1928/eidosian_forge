import collections
from tensorflow.python import pywrap_tfe as pywrap_tfe
from tensorflow.python.eager import context as _context
from tensorflow.python.eager import core as _core
from tensorflow.python.eager import execute as _execute
from tensorflow.python.framework import dtypes as _dtypes
from tensorflow.security.fuzzing.py import annotation_types as _atypes
from tensorflow.python.framework import op_def_registry as _op_def_registry
from tensorflow.python.framework import ops as _ops
from tensorflow.python.framework import op_def_library as _op_def_library
from tensorflow.python.util.deprecation import deprecated_endpoints
from tensorflow.python.util import dispatch as _dispatch
from tensorflow.python.util.tf_export import tf_export
from typing import TypeVar, List
def strided_slice_assign_eager_fallback(ref: _atypes.TensorFuzzingAnnotation[TV_StridedSliceAssign_T], begin: _atypes.TensorFuzzingAnnotation[TV_StridedSliceAssign_Index], end: _atypes.TensorFuzzingAnnotation[TV_StridedSliceAssign_Index], strides: _atypes.TensorFuzzingAnnotation[TV_StridedSliceAssign_Index], value: _atypes.TensorFuzzingAnnotation[TV_StridedSliceAssign_T], begin_mask: int, end_mask: int, ellipsis_mask: int, new_axis_mask: int, shrink_axis_mask: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_StridedSliceAssign_T]:
    raise RuntimeError("strided_slice_assign op does not support eager execution. Arg 'output_ref' is a ref.")