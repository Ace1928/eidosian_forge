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
def prelinearize_tuple(inputs, shapes, layouts=[], name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """An op which linearizes multiple Tensor values to an opaque variant tensor.

  Args:
    inputs: A list of `Tensor` objects.
      A list of tensors that will be provided using the infeed mechanism.
    shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`).
      The shapes of each tensor in `inputs`.
    layouts: An optional list of `ints`. Defaults to `[]`.
      A vector holding the requested layout in minor-to-major sequence for all the
      tuple shapes in the order the shapes appear in the "shapes" input. The layout
      elements for a sub-shape can be set to -1 in which case the corresponding layout
      will be computed by the infeed operation.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'PrelinearizeTuple', name, inputs, 'shapes', shapes, 'layouts', layouts)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return prelinearize_tuple_eager_fallback(inputs, shapes=shapes, layouts=layouts, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(shapes, (list, tuple)):
        raise TypeError("Expected list for 'shapes' argument to 'prelinearize_tuple' Op, not %r." % shapes)
    shapes = [_execute.make_shape(_s, 'shapes') for _s in shapes]
    if layouts is None:
        layouts = []
    if not isinstance(layouts, (list, tuple)):
        raise TypeError("Expected list for 'layouts' argument to 'prelinearize_tuple' Op, not %r." % layouts)
    layouts = [_execute.make_int(_i, 'layouts') for _i in layouts]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('PrelinearizeTuple', inputs=inputs, shapes=shapes, layouts=layouts, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('dtypes', _op.get_attr('dtypes'), 'shapes', _op.get_attr('shapes'), 'layouts', _op.get_attr('layouts'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('PrelinearizeTuple', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result