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
def boosted_trees_create_ensemble(tree_ensemble_handle: _atypes.TensorFuzzingAnnotation[_atypes.Resource], stamp_token: _atypes.TensorFuzzingAnnotation[_atypes.Int64], tree_ensemble_serialized: _atypes.TensorFuzzingAnnotation[_atypes.String], name=None):
    """Creates a tree ensemble model and returns a handle to it.

  Args:
    tree_ensemble_handle: A `Tensor` of type `resource`.
      Handle to the tree ensemble resource to be created.
    stamp_token: A `Tensor` of type `int64`.
      Token to use as the initial value of the resource stamp.
    tree_ensemble_serialized: A `Tensor` of type `string`.
      Serialized proto of the tree ensemble.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
    _ctx = _context._context or _context.context()
    tld = _ctx._thread_local_data
    if tld.is_eager:
        try:
            _result = pywrap_tfe.TFE_Py_FastPathExecute(_ctx, 'BoostedTreesCreateEnsemble', name, tree_ensemble_handle, stamp_token, tree_ensemble_serialized)
            return _result
        except _core._NotOkStatusException as e:
            _ops.raise_from_not_ok_status(e, name)
        except _core._FallbackException:
            pass
        try:
            return boosted_trees_create_ensemble_eager_fallback(tree_ensemble_handle, stamp_token, tree_ensemble_serialized, name=name, ctx=_ctx)
        except _core._SymbolicException:
            pass
    _, _, _op, _outputs = _op_def_library._apply_op_helper('BoostedTreesCreateEnsemble', tree_ensemble_handle=tree_ensemble_handle, stamp_token=stamp_token, tree_ensemble_serialized=tree_ensemble_serialized, name=name)
    return _op