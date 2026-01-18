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
def n_ints_in_eager_fallback(a: List[_atypes.TensorFuzzingAnnotation[_atypes.Int32]], name, ctx):
    if not isinstance(a, (list, tuple)):
        raise TypeError("Expected list for 'a' argument to 'n_ints_in' Op, not %r." % a)
    _attr_N = len(a)
    a = _ops.convert_n_to_tensor(a, _dtypes.int32)
    _inputs_flat = list(a)
    _attrs = ('N', _attr_N)
    _result = _execute.execute(b'NIntsIn', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result