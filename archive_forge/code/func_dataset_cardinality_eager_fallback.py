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
def dataset_cardinality_eager_fallback(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], cardinality_options: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Int64]:
    if cardinality_options is None:
        cardinality_options = ''
    cardinality_options = _execute.make_str(cardinality_options, 'cardinality_options')
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    _inputs_flat = [input_dataset]
    _attrs = ('cardinality_options', cardinality_options)
    _result = _execute.execute(b'DatasetCardinality', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('DatasetCardinality', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result