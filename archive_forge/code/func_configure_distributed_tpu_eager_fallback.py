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
def configure_distributed_tpu_eager_fallback(embedding_config: str, tpu_embedding_config: str, is_global_init: bool, enable_whole_mesh_compilations: bool, compilation_failure_closes_chips: bool, tpu_cancellation_closes_chips: int, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    if embedding_config is None:
        embedding_config = ''
    embedding_config = _execute.make_str(embedding_config, 'embedding_config')
    if tpu_embedding_config is None:
        tpu_embedding_config = ''
    tpu_embedding_config = _execute.make_str(tpu_embedding_config, 'tpu_embedding_config')
    if is_global_init is None:
        is_global_init = False
    is_global_init = _execute.make_bool(is_global_init, 'is_global_init')
    if enable_whole_mesh_compilations is None:
        enable_whole_mesh_compilations = False
    enable_whole_mesh_compilations = _execute.make_bool(enable_whole_mesh_compilations, 'enable_whole_mesh_compilations')
    if compilation_failure_closes_chips is None:
        compilation_failure_closes_chips = True
    compilation_failure_closes_chips = _execute.make_bool(compilation_failure_closes_chips, 'compilation_failure_closes_chips')
    if tpu_cancellation_closes_chips is None:
        tpu_cancellation_closes_chips = 0
    tpu_cancellation_closes_chips = _execute.make_int(tpu_cancellation_closes_chips, 'tpu_cancellation_closes_chips')
    _inputs_flat = []
    _attrs = ('embedding_config', embedding_config, 'tpu_embedding_config', tpu_embedding_config, 'is_global_init', is_global_init, 'enable_whole_mesh_compilations', enable_whole_mesh_compilations, 'compilation_failure_closes_chips', compilation_failure_closes_chips, 'tpu_cancellation_closes_chips', tpu_cancellation_closes_chips)
    _result = _execute.execute(b'ConfigureDistributedTPU', 1, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('ConfigureDistributedTPU', _inputs_flat, _attrs, _result)
    _result, = _result
    return _result