import collections
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import cond
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import tensor_array_ops
def empty_list_of_elements_like_x():
    tensor_x = ops.convert_to_tensor(x)
    return list_ops.empty_tensor_list(element_shape=array_ops.shape(tensor_x), element_dtype=tensor_x.dtype)