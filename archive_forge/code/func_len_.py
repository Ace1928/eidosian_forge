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
def len_(s):
    len_override = registry_lookup(len_registry, s)
    if len_override is not None:
        return len_override(s)
    if tensors.is_tensor_array(s):
        return _tf_tensor_array_len(s)
    elif tensors.is_tensor_list(s):
        return _tf_tensor_list_len(s)
    elif tensor_util.is_tf_type(s):
        return _tf_tensor_len(s)
    return _py_len(s)