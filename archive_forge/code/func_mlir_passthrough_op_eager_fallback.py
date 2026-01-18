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
def mlir_passthrough_op_eager_fallback(inputs, mlir_module: str, Toutputs, name, ctx):
    mlir_module = _execute.make_str(mlir_module, 'mlir_module')
    if not isinstance(Toutputs, (list, tuple)):
        raise TypeError("Expected list for 'Toutputs' argument to 'mlir_passthrough_op' Op, not %r." % Toutputs)
    Toutputs = [_execute.make_type(_t, 'Toutputs') for _t in Toutputs]
    _attr_Tinputs, inputs = _execute.convert_to_mixed_eager_tensors(inputs, ctx)
    _inputs_flat = list(inputs)
    _attrs = ('mlir_module', mlir_module, 'Tinputs', _attr_Tinputs, 'Toutputs', Toutputs)
    _result = _execute.execute(b'MlirPassthroughOp', len(Toutputs), inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('MlirPassthroughOp', _inputs_flat, _attrs, _result)
    return _result