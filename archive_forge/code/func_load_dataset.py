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
def load_dataset(path: _atypes.TensorFuzzingAnnotation[_atypes.String], reader_func_other_args, output_types, output_shapes, reader_func, compression: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """TODO: add doc.

  Args:
    path: A `Tensor` of type `string`.
    reader_func_other_args: A list of `Tensor` objects.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    reader_func: A function decorated with @Defun.
    compression: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'LoadDataset', name, path, reader_func_other_args, 'output_types', output_types, 'output_shapes', output_shapes, 'compression', compression, 'reader_func', reader_func)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return load_dataset_eager_fallback(path, reader_func_other_args, output_types=output_types, output_shapes=output_shapes, compression=compression, reader_func=reader_func, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'load_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'load_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if compression is None:
        compression = ''
    compression = _execute.make_str(compression, 'compression')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('LoadDataset', path=path, reader_func_other_args=reader_func_other_args, output_types=output_types, output_shapes=output_shapes, reader_func=reader_func, compression=compression, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'), 'compression', _op.get_attr('compression'), 'reader_func', _op.get_attr('reader_func'), 'Treader_func_args', _op.get_attr('Treader_func_args'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('LoadDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result