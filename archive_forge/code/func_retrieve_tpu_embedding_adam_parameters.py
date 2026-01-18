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
def retrieve_tpu_embedding_adam_parameters(num_shards: int, shard_id: int, table_id: int=-1, table_name: str='', config: str='', name=None):
    """Retrieve ADAM embedding parameters.

  An op that retrieves optimization parameters from embedding to host
  memory. Must be preceded by a ConfigureTPUEmbeddingHost op that sets up
  the correct embedding table configuration. For example, this op is
  used to retrieve updated parameters before saving a checkpoint.

  Args:
    num_shards: An `int`.
    shard_id: An `int`.
    table_id: An optional `int`. Defaults to `-1`.
    table_name: An optional `string`. Defaults to `""`.
    config: An optional `string`. Defaults to `""`.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (parameters, momenta, velocities).

    parameters: A `Tensor` of type `float32`.
    momenta: A `Tensor` of type `float32`.
    velocities: A `Tensor` of type `float32`.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'RetrieveTPUEmbeddingADAMParameters', name, 'table_id', table_id, 'table_name', table_name, 'num_shards', num_shards, 'shard_id', shard_id, 'config', config)
            _result = _RetrieveTPUEmbeddingADAMParametersOutput._make(_result)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return retrieve_tpu_embedding_adam_parameters_eager_fallback(table_id=table_id, table_name=table_name, num_shards=num_shards, shard_id=shard_id, config=config, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    num_shards = _execute.make_int(num_shards, 'num_shards')
    shard_id = _execute.make_int(shard_id, 'shard_id')
    if table_id is None:
        table_id = -1
    table_id = _execute.make_int(table_id, 'table_id')
    if table_name is None:
        table_name = ''
    table_name = _execute.make_str(table_name, 'table_name')
    if config is None:
        config = ''
    config = _execute.make_str(config, 'config')
    _, _, _op, _outputs = _op_def_library._apply_op_helper('RetrieveTPUEmbeddingADAMParameters', num_shards=num_shards, shard_id=shard_id, table_id=table_id, table_name=table_name, config=config, name=name)
    _result = _outputs[:]
    if _execute.must_record_gradient():
        _attrs = ('table_id', _op._get_attr_int('table_id'), 'table_name', _op.get_attr('table_name'), 'num_shards', _op._get_attr_int('num_shards'), 'shard_id', _op._get_attr_int('shard_id'), 'config', _op.get_attr('config'))
        _inputs_flat = _op.inputs
        _execute.record_gradient('RetrieveTPUEmbeddingADAMParameters', _inputs_flat, _attrs, _result)
    _result = _RetrieveTPUEmbeddingADAMParametersOutput._make(_result)
    return _result