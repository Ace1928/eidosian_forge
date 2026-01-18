import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework.weak_tensor import WeakTensor
def get_test_input_for_op(val, dtype):
    """Returns a list containing all the possible inputs with a given dtype.

  Args:
    val: value to convert to test input.
    dtype: a tuple of format (tf.Dtype, bool) where the bool value represents
      whether the dtype is "weak" or not.

  Returns:
    A list of all possible inputs given a value and a dtype.
  """
    python_inferred_types = {(dtypes.int32, True): 1, (dtypes.float32, True): 1.0, (dtypes.complex128, True): 1j}
    dtype, weak = dtype
    inputs = []
    if weak:
        inputs.append(convert_to_input_type(val, 'WeakTensor', dtype))
        if dtype in python_inferred_types:
            val_in_dtype = val * python_inferred_types[dtype]
            inputs.append(val_in_dtype)
            inputs.append(convert_to_input_type(val_in_dtype, 'Tensor', None))
    else:
        inputs.append(convert_to_input_type(val, 'Tensor', dtype))
        inputs.append(convert_to_input_type(val, 'NumPy', dtype))
    return inputs