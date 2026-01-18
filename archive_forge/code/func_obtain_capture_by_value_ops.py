import queue
from tensorflow.python.framework import dtypes
def obtain_capture_by_value_ops(dataset):
    """Given an input dataset, finds all allowlisted ops used for construction.

  Allowlisted ops are stateful ops which are known to be safe to capture by
  value.

  Args:
    dataset: Dataset to find allowlisted stateful ops for.

  Returns:
    A list of variant_tensor producing dataset ops used to construct this
    dataset.
  """

    def capture_by_value(op):
        return op.outputs[0].dtype in TENSOR_TYPES_ALLOWLIST or op.type in OP_TYPES_ALLOWLIST
    return _traverse(dataset, capture_by_value)