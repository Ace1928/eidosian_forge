from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import gen_ragged_math_ops
from tensorflow.python.ops import math_ops
def lengths_to_splits(lengths):
    """Returns splits corresponding to the given lengths."""
    return array_ops.concat([[0], math_ops.cumsum(lengths)], axis=-1)