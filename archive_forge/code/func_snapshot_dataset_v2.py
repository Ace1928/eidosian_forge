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
def snapshot_dataset_v2(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], path: _atypes.TensorFuzzingAnnotation[_atypes.String], reader_func_other_args, shard_func_other_args, output_types, output_shapes, reader_func, shard_func, compression: str='', reader_prefix: str='', writer_prefix: str='', hash_valid: bool=False, hash: int=0, metadata: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
    """Creates a dataset that will write to / read from a snapshot.

  This dataset attempts to determine whether a valid snapshot exists at the
  `snapshot_path`, and reads from the snapshot in lieu of using `input_dataset`.
  If not, it will run the preprocessing pipeline as usual, and write out a
  snapshot of the data processed for future use.

  Args:
    input_dataset: A `Tensor` of type `variant`.
      A variant tensor representing the input dataset.
    path: A `Tensor` of type `string`.
      The path we should write snapshots to / read snapshots from.
    reader_func_other_args: A list of `Tensor` objects.
    shard_func_other_args: A list of `Tensor` objects.
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    reader_func: A function decorated with @Defun.
      Optional. A function to control how to read data from snapshot shards.
    shard_func: A function decorated with @Defun.
      Optional. A function to control how to shard data when writing a snapshot.
    compression: An optional `string`. Defaults to `""`.
      The type of compression to be applied to the saved snapshot files.
    reader_prefix: An optional `string`. Defaults to `""`.
    writer_prefix: An optional `string`. Defaults to `""`.
    hash_valid: An optional `bool`. Defaults to `False`.
    hash: An optional `int`. Defaults to `0`.
    metadata: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SnapshotDatasetV2', name, input_dataset, path, reader_func_other_args, shard_func_other_args, 'output_types', output_types, 'output_shapes', output_shapes, 'compression', compression, 'reader_prefix', reader_prefix, 'writer_prefix', writer_prefix, 'hash_valid', hash_valid, 'hash', hash, 'reader_func', reader_func, 'shard_func', shard_func, 'metadata', metadata)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return snapshot_dataset_v2_eager_fallback(input_dataset, path, reader_func_other_args, shard_func_other_args, output_types=output_types, output_shapes=output_shapes, compression=compression, reader_prefix=reader_prefix, writer_prefix=writer_prefix, hash_valid=hash_valid, hash=hash, reader_func=reader_func, shard_func=shard_func, metadata=metadata, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'snapshot_dataset_v2' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'snapshot_dataset_v2' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if compression is None:
        compression = ''
    compression = _execute.make_str(compression, 'compression')
    if reader_prefix is None:
        reader_prefix = ''
    reader_prefix = _execute.make_str(reader_prefix, 'reader_prefix')
    if writer_prefix is None:
        writer_prefix = ''
    writer_prefix = _execute.make_str(writer_prefix, 'writer_prefix')
    if hash_valid is None:
        hash_valid = False
    hash_valid = _execute.make_bool(hash_valid, 'hash_valid')
    if hash is None:
        hash = 0
    hash = _execute.make_int(hash, 'hash')
    if metadata is None:
        metadata = ''
    metadata = _execute.make_str(metadata, 'metadata')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SnapshotDatasetV2', input_dataset=input_dataset, path=path, reader_func_other_args=reader_func_other_args, shard_func_other_args=shard_func_other_args, output_types=output_types, output_shapes=output_shapes, reader_func=reader_func, shard_func=shard_func, compression=compression, reader_prefix=reader_prefix, writer_prefix=writer_prefix, hash_valid=hash_valid, hash=hash, metadata=metadata, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'), 'compression', _op.get_attr('compression'), 'reader_prefix', _op.get_attr('reader_prefix'), 'writer_prefix', _op.get_attr('writer_prefix'), 'hash_valid', _op._get_attr_bool('hash_valid'), 'hash', _op._get_attr_int('hash'), 'reader_func', _op.get_attr('reader_func'), 'shard_func', _op.get_attr('shard_func'), 'Treader_func_args', _op.get_attr('Treader_func_args'), 'Tshard_func_args', _op.get_attr('Tshard_func_args'), 'metadata', _op.get_attr('metadata'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SnapshotDatasetV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result