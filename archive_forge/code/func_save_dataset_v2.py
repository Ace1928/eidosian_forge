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
def save_dataset_v2(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], path: _atypes.TensorFuzzingAnnotation[_atypes.String], shard_func_other_args, shard_func, output_types, output_shapes, compression: str='', use_shard_func: bool=True, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """TODO: add doc.

  Args:
    input_dataset: A `Tensor` of type `variant`.
    path: A `Tensor` of type `string`.
    shard_func_other_args: A list of `Tensor` objects.
    shard_func: A function decorated with @Defun.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    compression: An optional `string`. Defaults to `""`.
    use_shard_func: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SaveDatasetV2', name, input_dataset, path, shard_func_other_args, 'compression', compression, 'shard_func', shard_func, 'use_shard_func', use_shard_func, 'output_types', output_types, 'output_shapes', output_shapes)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return save_dataset_v2_eager_fallback(input_dataset, path, shard_func_other_args, compression=compression, shard_func=shard_func, use_shard_func=use_shard_func, output_types=output_types, output_shapes=output_shapes, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'save_dataset_v2' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'save_dataset_v2' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if compression is None:
        compression = ''
    compression = _execute.make_str(compression, 'compression')
    if use_shard_func is None:
        use_shard_func = True
    use_shard_func = _execute.make_bool(use_shard_func, 'use_shard_func')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SaveDatasetV2', input_dataset=input_dataset, path=path, shard_func_other_args=shard_func_other_args, shard_func=shard_func, output_types=output_types, output_shapes=output_shapes, compression=compression, use_shard_func=use_shard_func, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('compression', _op.get_attr('compression'), 'shard_func', _op.get_attr('shard_func'), 'use_shard_func', _op._get_attr_bool('use_shard_func'), 'Tshard_func_args', _op.get_attr('Tshard_func_args'), 'output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SaveDatasetV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result