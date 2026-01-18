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
def tensor_slice_dataset(components, output_shapes, is_files: bool=False, metadata: str='', replicate_on_split: bool=False, name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Creates a dataset that emits each dim-0 slice of `components` once.

  Args:
    components: A list of `Tensor` objects.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    is_files: An optional `bool`. Defaults to `False`.
    metadata: An optional `string`. Defaults to `""`.
    replicate_on_split: An optional `bool`. Defaults to `False`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'TensorSliceDataset', name, components, 'output_shapes', output_shapes, 'is_files', is_files, 'metadata', metadata, 'replicate_on_split', replicate_on_split)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return tensor_slice_dataset_eager_fallback(components, output_shapes=output_shapes, is_files=is_files, metadata=metadata, replicate_on_split=replicate_on_split, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'tensor_slice_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if is_files is None:
        is_files = False
    is_files = _execute.make_bool(is_files, 'is_files')
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    if replicate_on_split is None:
        replicate_on_split = False
    replicate_on_split = _execute.make_bool(replicate_on_split, 'replicate_on_split')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('TensorSliceDataset', components=components, output_shapes=output_shapes, is_files=is_files, metadata=metadata, replicate_on_split=replicate_on_split, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('Toutput_types', _op.get_attr('Toutput_types'), 'output_shapes', _op.get_attr('output_shapes'), 'is_files', _op._get_attr_bool('is_files'), 'metadata', _op.get_attr('metadata'), 'replicate_on_split', _op._get_attr_bool('replicate_on_split'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('TensorSliceDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result