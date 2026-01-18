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
def recv_tpu_embedding_activations_eager_fallback(num_outputs: int, config: str, name, ctx):
    num_outputs = _execute.make_int(num_outputs, 'num_outputs')
    config = _execute.make_str(config, 'config')
    _inputs_flat = []
    _attrs = ('num_outputs', num_outputs, 'config', config)
    _result = _execute.execute(b'RecvTPUEmbeddingActivations', num_outputs, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    if _execute.must_record_gradient():
        _execute.record_gradient('RecvTPUEmbeddingActivations', _inputs_flat, _attrs, _result)
    return _result