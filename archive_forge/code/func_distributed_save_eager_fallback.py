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
def distributed_save_eager_fallback(dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], directory: _atypes.TensorFuzzingAnnotation[_atypes.String], address: _atypes.TensorFuzzingAnnotation[_atypes.String], metadata: str, name, ctx):
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    dataset = _ops.convert_to_tensor(dataset, _dtypes.variant)
    directory = _ops.convert_to_tensor(directory, _dtypes.string)
    address = _ops.convert_to_tensor(address, _dtypes.string)
    _inputs_flat = [dataset, directory, address]
    _attrs = ('metadata', metadata)
    _result = _execute.execute(b'DistributedSave', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result