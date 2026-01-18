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
def tensor_list_concat_eager_fallback(input_handle: _atypes.TensorFuzzingAnnotation[_atypes.Variant], element_dtype: TV_TensorListConcat_element_dtype, element_shape, name, ctx):
    element_dtype = _execute.make_type(element_dtype, 'element_dtype')
    if element_shape is None:
        element_shape = None
    element_shape = _execute.make_shape(element_shape, 'element_shape')
    input_handle = _ops.convert_to_tensor(input_handle, _dtypes.variant)
    _inputs_flat = [input_handle]
    _attrs = ('element_dtype', element_dtype, 'element_shape', element_shape)
    _result = _execute.execute(b'TensorListConcat', 2, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('TensorListConcat', _inputs_flat, _attrs, _result)
    _result = _TensorListConcatOutput._make(_result)
    return _result