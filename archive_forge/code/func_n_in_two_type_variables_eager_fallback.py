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
def n_in_two_type_variables_eager_fallback(a: List[_atypes.TensorFuzzingAnnotation[TV_NInTwoTypeVariables_S]], b: List[_atypes.TensorFuzzingAnnotation[TV_NInTwoTypeVariables_T]], name, ctx):
    if not isinstance(a, (list, tuple)):
        raise TypeError("Expected list for 'a' argument to 'n_in_two_type_variables' Op, not %r." % a)
    _attr_N = len(a)
    if not isinstance(b, (list, tuple)):
        raise TypeError("Expected list for 'b' argument to 'n_in_two_type_variables' Op, not %r." % b)
    if len(b) != _attr_N:
        raise ValueError("List argument 'b' to 'n_in_two_type_variables' Op with length %d must match length %d of argument 'a'." % (len(b), _attr_N))
    _attr_S, a = _execute.args_to_matching_eager(list(a), ctx, [])
    _attr_T, b = _execute.args_to_matching_eager(list(b), ctx, [])
    _inputs_flat = list(a) + list(b)
    _attrs = ('S', _attr_S, 'T', _attr_T, 'N', _attr_N)
    _result = _execute.execute(b'NInTwoTypeVariables', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result