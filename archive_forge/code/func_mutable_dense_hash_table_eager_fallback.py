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
def mutable_dense_hash_table_eager_fallback(empty_key: _atypes.TensorFuzzingAnnotation[TV_MutableDenseHashTable_key_dtype], value_dtype: TV_MutableDenseHashTable_value_dtype, container: str, shared_name: str, use_node_name_sharing: bool, value_shape, initial_num_buckets: int, max_load_factor: float, name, ctx) -> _atypes.TensorFuzzingAnnotation[_atypes.String]:
    raise RuntimeError("mutable_dense_hash_table op does not support eager execution. Arg 'table_handle' is a ref.")