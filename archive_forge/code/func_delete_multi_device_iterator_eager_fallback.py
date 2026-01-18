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
def delete_multi_device_iterator_eager_fallback(multi_device_iterator: _atypes.TensorFuzzingAnnotation[_atypes.Resource], iterators: List[_atypes.TensorFuzzingAnnotation[_atypes.Resource]], deleter: _atypes.TensorFuzzingAnnotation[_atypes.Variant], name, ctx):
    if not isinstance(iterators, (list, tuple)):
        raise TypeError("Expected list for 'iterators' argument to 'delete_multi_device_iterator' Op, not %r." % iterators)
    _attr_N = len(iterators)
    multi_device_iterator = _ops.convert_to_tensor(multi_device_iterator, _dtypes.resource)
    iterators = _ops.convert_n_to_tensor(iterators, _dtypes.resource)
    deleter = _ops.convert_to_tensor(deleter, _dtypes.variant)
    _inputs_flat = [multi_device_iterator] + list(iterators) + [deleter]
    _attrs = ('N', _attr_N)
    _result = _execute.execute(b'DeleteMultiDeviceIterator', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result