from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import math_ops
def assert_splits_match(nested_splits_lists):
    """Checks that the given splits lists are identical.

  Performs static tests to ensure that the given splits lists are identical,
  and returns a list of control dependency op tensors that check that they are
  fully identical.

  Args:
    nested_splits_lists: A list of nested_splits_lists, where each split_list is
      a list of `splits` tensors from a `RaggedTensor`, ordered from outermost
      ragged dimension to innermost ragged dimension.

  Returns:
    A list of control dependency op tensors.
  Raises:
    ValueError: If the splits are not identical.
  """
    error_msg = 'Inputs must have identical ragged splits'
    for splits_list in nested_splits_lists:
        if len(splits_list) != len(nested_splits_lists[0]):
            raise ValueError(error_msg)
    return [check_ops.assert_equal(s1, s2, message=error_msg) for splits_list in nested_splits_lists[1:] for s1, s2 in zip(nested_splits_lists[0], splits_list)]