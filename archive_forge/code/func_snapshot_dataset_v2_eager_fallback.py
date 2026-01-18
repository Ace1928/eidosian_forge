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
def snapshot_dataset_v2_eager_fallback(input_dataset: _atypes.TensorFuzzingAnnotation[_atypes.Variant], path: _atypes.TensorFuzzingAnnotation[_atypes.String], reader_func_other_args, shard_func_other_args, output_types, output_shapes, reader_func, shard_func, compression: str, reader_prefix: str, writer_prefix: str, hash_valid: bool, hash: int, metadata: str, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.Variant]:
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
    _attr_Treader_func_args, reader_func_other_args = _execute.convert_to_mixed_eager_tensors(reader_func_other_args, ctx)
    _attr_Tshard_func_args, shard_func_other_args = _execute.convert_to_mixed_eager_tensors(shard_func_other_args, ctx)
    input_dataset = _ops.convert_to_tensor(input_dataset, _dtypes.variant)
    path = _ops.convert_to_tensor(path, _dtypes.string)
    _inputs_flat = [input_dataset, path] + list(reader_func_other_args) + list(shard_func_other_args)
    _attrs = ('output_types', output_types, 'output_shapes', output_shapes, 'compression', compression, 'reader_prefix', reader_prefix, 'writer_prefix', writer_prefix, 'hash_valid', hash_valid, 'hash', hash, 'reader_func', reader_func, 'shard_func', shard_func, 'Treader_func_args', _attr_Treader_func_args, 'Tshard_func_args', _attr_Tshard_func_args, 'metadata', metadata)
    _result = _execute.execute(b'SnapshotDatasetV2', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('SnapshotDatasetV2', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result