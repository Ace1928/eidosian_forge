import inspect
from tensorflow.python.autograph.utils import tensors
from tensorflow.python.autograph.utils import type_registry
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import control_flow_assert
from tensorflow.python.ops import gen_parsing_ops
from tensorflow.python.ops import gen_string_ops
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
def map_(fn, *iterables):
    map_fn = _py_map
    for x in iterables:
        map_override = registry_lookup(map_registry, x)
        if map_override is None or (map_fn != _py_map and map_override != map_fn):
            map_fn = _py_map
            break
        map_fn = map_override
    return map_fn(fn, *iterables)