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
def snapshot_dataset(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], path: _atypes.TensorFuzzingAnnotation[_atypes.String], output_types, output_shapes, compression: str='', reader_path_prefix: str='', writer_path_prefix: str='', shard_size_bytes: int=10737418240, pending_snapshot_expiry_seconds: int=86400, num_reader_threads: int=1, reader_buffer_size: int=1, num_writer_threads: int=1, writer_buffer_size: int=1, shuffle_on_read: bool=False, seed: int=0, seed2: int=0, mode: str='auto', snapshot_name: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
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
    output_types: A list of `tf.DTypes` that has length `>= 1`.
    output_shapes: A list of shapes (each a `tf.TensorShape` or list of `ints`) that has length `>= 1`.
    compression: An optional `string`. Defaults to `""`.
    reader_path_prefix: An optional `string`. Defaults to `""`.
    writer_path_prefix: An optional `string`. Defaults to `""`.
    shard_size_bytes: An optional `int`. Defaults to `10737418240`.
    pending_snapshot_expiry_seconds: An optional `int`. Defaults to `86400`.
    num_reader_threads: An optional `int`. Defaults to `1`.
    reader_buffer_size: An optional `int`. Defaults to `1`.
    num_writer_threads: An optional `int`. Defaults to `1`.
    writer_buffer_size: An optional `int`. Defaults to `1`.
    shuffle_on_read: An optional `bool`. Defaults to `False`.
    seed: An optional `int`. Defaults to `0`.
    seed2: An optional `int`. Defaults to `0`.
    mode: An optional `string`. Defaults to `"auto"`.
    snapshot_name: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `variant`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'SnapshotDataset', name, input_dataset, path, 'output_types', output_types, 'output_shapes', output_shapes, 'compression', compression, 'reader_path_prefix', reader_path_prefix, 'writer_path_prefix', writer_path_prefix, 'shard_size_bytes', shard_size_bytes, 'pending_snapshot_expiry_seconds', pending_snapshot_expiry_seconds, 'num_reader_threads', num_reader_threads, 'reader_buffer_size', reader_buffer_size, 'num_writer_threads', num_writer_threads, 'writer_buffer_size', writer_buffer_size, 'shuffle_on_read', shuffle_on_read, 'seed', seed, 'seed2', seed2, 'mode', mode, 'snapshot_name', snapshot_name)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return snapshot_dataset_eager_fallback(input_dataset, path, output_types=output_types, output_shapes=output_shapes, compression=compression, reader_path_prefix=reader_path_prefix, writer_path_prefix=writer_path_prefix, shard_size_bytes=shard_size_bytes, pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds, num_reader_threads=num_reader_threads, reader_buffer_size=reader_buffer_size, num_writer_threads=num_writer_threads, writer_buffer_size=writer_buffer_size, shuffle_on_read=shuffle_on_read, seed=seed, seed2=seed2, mode=mode, snapshot_name=snapshot_name, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    if not isinstance(output_types, (list, tuple)):
        raise TypeError("Expected list for 'output_types' argument to 'snapshot_dataset' Op, not %r." % output_types)
    output_types = [_execute.make_type(_t, 'output_types') for _t in output_types]
    if not isinstance(output_shapes, (list, tuple)):
        raise TypeError("Expected list for 'output_shapes' argument to 'snapshot_dataset' Op, not %r." % output_shapes)
    output_shapes = [_execute.make_shape(_s, 'output_shapes') for _s in output_shapes]
    if compression is None:
        compression = ''
    compression = _execute.make_str(compression, 'compression')
    if reader_path_prefix is None:
        reader_path_prefix = ''
    reader_path_prefix = _execute.make_str(reader_path_prefix, 'reader_path_prefix')
    if writer_path_prefix is None:
        writer_path_prefix = ''
    writer_path_prefix = _execute.make_str(writer_path_prefix, 'writer_path_prefix')
    if shard_size_bytes is None:
        shard_size_bytes = 10737418240
    shard_size_bytes = _execute.make_int(shard_size_bytes, 'shard_size_bytes')
    if pending_snapshot_expiry_seconds is None:
        pending_snapshot_expiry_seconds = 86400
    pending_snapshot_expiry_seconds = _execute.make_int(pending_snapshot_expiry_seconds, 'pending_snapshot_expiry_seconds')
    if num_reader_threads is None:
        num_reader_threads = 1
    num_reader_threads = _execute.make_int(num_reader_threads, 'num_reader_threads')
    if reader_buffer_size is None:
        reader_buffer_size = 1
    reader_buffer_size = _execute.make_int(reader_buffer_size, 'reader_buffer_size')
    if num_writer_threads is None:
        num_writer_threads = 1
    num_writer_threads = _execute.make_int(num_writer_threads, 'num_writer_threads')
    if writer_buffer_size is None:
        writer_buffer_size = 1
    writer_buffer_size = _execute.make_int(writer_buffer_size, 'writer_buffer_size')
    if shuffle_on_read is None:
        shuffle_on_read = False
    shuffle_on_read = _execute.make_bool(shuffle_on_read, 'shuffle_on_read')
    if seed is None:
        seed = 0
    seed = _execute.make_int(seed, 'seed')
    if seed2 is None:
        seed2 = 0
    seed2 = _execute.make_int(seed2, 'seed2')
    if mode is None:
        mode = 'auto'
    mode = _execute.make_str(mode, 'mode')
    if snapshot_name is None:
        snapshot_name = ''
    snapshot_name = _execute.make_str(snapshot_name, 'snapshot_name')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('SnapshotDataset', input_dataset=input_dataset, path=path, output_types=output_types, output_shapes=output_shapes, compression=compression, reader_path_prefix=reader_path_prefix, writer_path_prefix=writer_path_prefix, shard_size_bytes=shard_size_bytes, pending_snapshot_expiry_seconds=pending_snapshot_expiry_seconds, num_reader_threads=num_reader_threads, reader_buffer_size=reader_buffer_size, num_writer_threads=num_writer_threads, writer_buffer_size=writer_buffer_size, shuffle_on_read=shuffle_on_read, seed=seed, seed2=seed2, mode=mode, snapshot_name=snapshot_name, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('output_types', _op.get_attr('output_types'), 'output_shapes', _op.get_attr('output_shapes'), 'compression', _op.get_attr('compression'), 'reader_path_prefix', _op.get_attr('reader_path_prefix'), 'writer_path_prefix', _op.get_attr('writer_path_prefix'), 'shard_size_bytes', _op._get_attr_int('shard_size_bytes'), 'pending_snapshot_expiry_seconds', _op._get_attr_int('pending_snapshot_expiry_seconds'), 'num_reader_threads', _op._get_attr_int('num_reader_threads'), 'reader_buffer_size', _op._get_attr_int('reader_buffer_size'), 'num_writer_threads', _op._get_attr_int('num_writer_threads'), 'writer_buffer_size', _op._get_attr_int('writer_buffer_size'), 'shuffle_on_read', _op._get_attr_bool('shuffle_on_read'), 'seed', _op._get_attr_int('seed'), 'seed2', _op._get_attr_int('seed2'), 'mode', _op.get_attr('mode'), 'snapshot_name', _op.get_attr('snapshot_name'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('SnapshotDataset', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result