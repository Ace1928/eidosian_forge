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
def ragged_tensor_to_tensor_eager_fallback(shape: _atypes.TensorFuzzingAnnotation[TV_RaggedTensorToTensor_Tshape], values: _atypes.TensorFuzzingAnnotation[TV_RaggedTensorToTensor_T], default_value: _atypes.TensorFuzzingAnnotation[TV_RaggedTensorToTensor_T], row_partition_tensors: List[_atypes.TensorFuzzingAnnotation[TV_RaggedTensorToTensor_Tindex]], row_partition_types, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_RaggedTensorToTensor_T]:
    if not isinstance(row_partition_tensors, (list, tuple)):
        raise TypeError("Expected list for 'row_partition_tensors' argument to 'ragged_tensor_to_tensor' Op, not %r." % row_partition_tensors)
    _attr_num_row_partition_tensors = len(row_partition_tensors)
    if not isinstance(row_partition_types, (list, tuple)):
        raise TypeError("Expected list for 'row_partition_types' argument to 'ragged_tensor_to_tensor' Op, not %r." % row_partition_types)
    row_partition_types = [_execute.make_str(_s, 'row_partition_types') for _s in row_partition_types]
    _attr_T, _inputs_T = _execute.args_to_matching_eager([values, default_value], ctx, [])
    values, default_value = _inputs_T
    _attr_Tindex, row_partition_tensors = _execute.args_to_matching_eager(list(row_partition_tensors), ctx, [_dtypes.int64, _dtypes.int32])
    _attr_Tshape, (shape,) = _execute.args_to_matching_eager([shape], ctx, [_dtypes.int64, _dtypes.int32])
    _inputs_flat = [shape, values, default_value] + list(row_partition_tensors)
    _attrs = ('T', _attr_T, 'Tindex', _attr_Tindex, 'Tshape', _attr_Tshape, 'num_row_partition_tensors', _attr_num_row_partition_tensors, 'row_partition_types', row_partition_types)
    _result = _execute.execute(b'RaggedTensorToTensor', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('RaggedTensorToTensor', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result