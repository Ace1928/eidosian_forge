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
def dataset_to_tf_record_eager_fallback(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], filename: _atypes.TensorFuzzingAnnotation[_atypes.String], compression_type: _atypes.TensorFuzzingAnnotation[_atypes.String], name, ctx):
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    filename = _ops.convert_to_tensor(filename, _dtypes.string)
    compression_type = _ops.convert_to_tensor(compression_type, _dtypes.string)
    _inputs_flat = [input_dataset, filename, compression_type]
    _attrs = None
    _result = _execute.execute(b'DatasetToTFRecord', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result