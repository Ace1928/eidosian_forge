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
def record_input(file_pattern: str, file_random_seed: int=301, file_shuffle_shift_ratio: float=0, file_buffer_size: int=10000, file_parallelism: int=16, batch_size: int=32, compression_type: str='', name=None) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    """Emits randomized records.

  Args:
    file_pattern: A `string`. Glob pattern for the data files.
    file_random_seed: An optional `int`. Defaults to `301`.
      Random seeds used to produce randomized records.
    file_shuffle_shift_ratio: An optional `float`. Defaults to `0`.
      Shifts the list of files after the list is randomly
      shuffled.
    file_buffer_size: An optional `int`. Defaults to `10000`.
      The randomization shuffling buffer.
    file_parallelism: An optional `int`. Defaults to `16`.
      How many sstables are opened and concurrently iterated over.
    batch_size: An optional `int`. Defaults to `32`. The batch size.
    compression_type: An optional `string`. Defaults to `""`.
      The type of compression for the file. Currently ZLIB and
      GZIP are supported. Defaults to none.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RecordInput', name, 'file_pattern', file_pattern, 'file_random_seed', file_random_seed, 'file_shuffle_shift_ratio', file_shuffle_shift_ratio, 'file_buffer_size', file_buffer_size, 'file_parallelism', file_parallelism, 'batch_size', batch_size, 'compression_type', compression_type)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return record_input_eager_fallback(file_pattern=file_pattern, file_random_seed=file_random_seed, file_shuffle_shift_ratio=file_shuffle_shift_ratio, file_buffer_size=file_buffer_size, file_parallelism=file_parallelism, batch_size=batch_size, compression_type=compression_type, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    file_pattern = _execute.make_str(file_pattern, 'file_pattern')
    if file_random_seed is None:
        file_random_seed = 301
    file_random_seed = _execute.make_int(file_random_seed, 'file_random_seed')
    if file_shuffle_shift_ratio is None:
        file_shuffle_shift_ratio = 0
    file_shuffle_shift_ratio = _execute.make_float(file_shuffle_shift_ratio, 'file_shuffle_shift_ratio')
    if file_buffer_size is None:
        file_buffer_size = 10000
    file_buffer_size = _execute.make_int(file_buffer_size, 'file_buffer_size')
    if file_parallelism is None:
        file_parallelism = 16
    file_parallelism = _execute.make_int(file_parallelism, 'file_parallelism')
    if batch_size is None:
        batch_size = 32
    batch_size = _execute.make_int(batch_size, 'batch_size')
    if compression_type is None:
        compression_type = ''
    compression_type = _execute.make_str(compression_type, 'compression_type')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RecordInput', file_pattern=file_pattern, file_random_seed=file_random_seed, file_shuffle_shift_ratio=file_shuffle_shift_ratio, file_buffer_size=file_buffer_size, file_parallelism=file_parallelism, batch_size=batch_size, compression_type=compression_type, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('file_pattern', _op.get_attr('file_pattern'), 'file_random_seed', _op._get_attr_int('file_random_seed'), 'file_shuffle_shift_ratio', _op.get_attr('file_shuffle_shift_ratio'), 'file_buffer_size', _op._get_attr_int('file_buffer_size'), 'file_parallelism', _op._get_attr_int('file_parallelism'), 'batch_size', _op._get_attr_int('batch_size'), 'compression_type', _op.get_attr('compression_type'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RecordInput', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result