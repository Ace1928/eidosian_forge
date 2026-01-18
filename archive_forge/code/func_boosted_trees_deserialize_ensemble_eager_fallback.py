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
def boosted_trees_deserialize_ensemble_eager_fallback(tree_ensemble_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], stamp_token: _atypes.TensorFuzzingAnnotation[_atypes.Int64], tree_ensemble_serialized: _atypes.TensorFuzzingAnnotation[_atypes.String], name, ctx):
    tree_ensemble_handle = _ops.convert_to_tensor(tree_ensemble_handle, _dtypes.resource)
    stamp_token = _ops.convert_to_tensor(stamp_token, _dtypes.int64)
    tree_ensemble_serialized = _ops.convert_to_tensor(tree_ensemble_serialized, _dtypes.string)
    _inputs_flat = [tree_ensemble_handle, stamp_token, tree_ensemble_serialized]
    _attrs = None
    _result = _execute.execute(b'BoostedTreesDeserializeEnsemble', 0, inputs=_inputs_flat, attrs=_attrs, ctx=ctx, name=name)
    _result = None
    return _result