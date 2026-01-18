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
def stateful_partitioned_call(args, Tout, f, config: str='', config_proto: str='', executor_type: str='', name=None):
    """returns `f(inputs)`, where `f`'s body is placed and partitioned.

  Args:
    args: A list of `Tensor` objects. A list of input tensors.
    Tout: A list of `tf.DTypes`. A list of output types.
    f: A function decorated with @Defun.
            A function that takes 'args', a list of tensors, and returns 'output',
            another list of tensors. Input and output types are specified by 'Tin'
            and 'Tout'. The function body of f will be placed and partitioned across
            devices, setting this op apart from the regular Call op. This op is
            stateful.
    config: An optional `string`. Defaults to `""`.
    config_proto: An optional `string`. Defaults to `""`.
    executor_type: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` objects of type `Tout`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'StatefulPartitionedCall', name, args, 'Tout', Tout, 'f', f, 'config', config, 'config_proto', config_proto, 'executor_type', executor_type)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return stateful_partitioned_call_eager_fallback(args, Tout=Tout, f=f, config=config, config_proto=config_proto, executor_type=executor_type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(Tout, (list, tuple)):
        raise TypeError("Expected list for 'Tout' argument to 'stateful_partitioned_call' Op, not %r." % Tout)
    Tout = [_execute.make_type(_t, 'Tout') for _t in Tout]
    if config is None:
        config = ''
    config = _execute.make_str(config, 'config')
    if config_proto is None:
        config_proto = ''
    config_proto = _execute.make_str(config_proto, 'config_proto')
    if executor_type is None:
        executor_type = ''
    executor_type = _execute.make_str(executor_type, 'executor_type')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('StatefulPartitionedCall', args=args, Tout=Tout, f=f, config=config, config_proto=config_proto, executor_type=executor_type, name=name)
    _result = _outputs[:]
    if not _result:
        return _op
    if _execute.must_record_gradient():
        _attrs = ('Tin', _op.get_attr('Tin'), 'Tout', _op.get_attr('Tout'), 'f', _op.get_attr('f'), 'config', _op.get_attr('config'), 'config_proto', _op.get_attr('config_proto'), 'executor_type', _op.get_attr('executor_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('StatefulPartitionedCall', _inputs_flat, _attrs, _result)
    return _result