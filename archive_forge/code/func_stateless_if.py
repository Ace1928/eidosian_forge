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
def stateless_if(cond: _atypes.TensorFuzzingAnnotation[TV_StatelessIf_Tcond], input, Tout, then_branch, else_branch, output_shapes=[], name=None):
    """output = cond ? then_branch(input) : else_branch(input)

  Args:
    cond: A `Tensor`.
            A Tensor. If the tensor is a scalar of non-boolean type, the
            scalar is converted to a boolean according to the
            following rule: if the scalar is a numerical value, non-zero means
            `True` and zero means False; if the scalar is a string, non-empty
            means `True` and empty means `False`. If the tensor is not a scalar,
            being empty means False and being non-empty means True.

            This should only be used when the if then/else body functions do not
            have stateful ops.
    input: A list of `Tensor` objects. A list of input tensors.
    Tout: A list of `tf.DTypes`. A list of output types.
    then_branch: A function decorated with @Defun.
            A function that takes 'inputs' and returns a list of tensors, whose
            types are the same as what else_branch returns.
    else_branch: A function decorated with @Defun.
          A function that takes 'inputs' and returns a list of tensors, whose
          types are the same as what then_branch returns.
    output_shapes: An optional list of shapes (each a `tf.TensorShape` or list of `ints`). Defaults to `[]`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StatelessIf', name, cond, input, 'Tout', Tout, 'then_branch', then_branch, 'else_branch', else_branch, 'output_shapes', output_shapes)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return stateless_if_eager_fallback(cond, input, Tout=Tout, then_branch=then_branch, else_branch=else_branch, output_shapes=output_shapes, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(Tout, (list, tuple)):
        raise TypeError("Expected list for 'Tout' argument to 'stateless_if' Op, not %r." % Tout)
    Tout = [_execute.make_type(_t, 'Tout') for _t in Tout]
    if output_shapes is None:
        output_shapes = []
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'stateless_if' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    _, _, _op, _outputs = _op_def_library._apply_op_helper('StatelessIf', cond=cond, input=input, Tout=Tout, then_branch=then_branch, else_branch=else_branch, output_shapes=output_shapes, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Tcond', _op._get_attr_type('Tcond'), 'Tin', _op.get_attr('Tin'), 'Tout', _op.get_attr('Tout'), 'then_branch', _op.get_attr('then_branch'), 'else_branch', _op.get_attr('else_branch'), 'output_shapes', _op.get_attr('output_shapes'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StatelessIf', _inputs_flat, _attrs, _result)
    return _result