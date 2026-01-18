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
def apply_ada_max_eager_fallback(var: _atypes.TensorFuzzingAnnotation[TV_ApplyAdaMax_T], m: _atypes.TensorFuzzingAnnotation[TV_ApplyAdaMax_T], v: _atypes.TensorFuzzingAnnotation[TV_ApplyAdaMax_T], beta1_power: _atypes.TensorFuzzingAnnotation[TV_ApplyAdaMax_T], lr: _atypes.TensorFuzzingAnnotation[TV_ApplyAdaMax_T], beta1: _atypes.TensorFuzzingAnnotation[TV_ApplyAdaMax_T], beta2: _atypes.TensorFuzzingAnnotation[TV_ApplyAdaMax_T], epsilon: _atypes.TensorFuzzingAnnotation[TV_ApplyAdaMax_T], grad: _atypes.TensorFuzzingAnnotation[TV_ApplyAdaMax_T], use_locking: bool, name, ctx) -> _atypes.TensorFuzzingAnnotation[TV_ApplyAdaMax_T]:
    raise RuntimeError("apply_ada_max op does not support eager execution. Arg 'out' is a ref.")