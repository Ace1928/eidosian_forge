import math
from tensorflow.python.keras import backend
from tensorflow.python.keras.utils.generic_utils import deserialize_keras_object
from tensorflow.python.keras.utils.generic_utils import serialize_keras_object
from tensorflow.python.ops import math_ops
def l1_l2(l1=0.01, l2=0.01):
    """Create a regularizer that applies both L1 and L2 penalties.

  The L1 regularization penalty is computed as:
  `loss = l1 * reduce_sum(abs(x))`

  The L2 regularization penalty is computed as:
  `loss = l2 * reduce_sum(square(x))`

  Args:
      l1: Float; L1 regularization factor.
      l2: Float; L2 regularization factor.

  Returns:
    An L1L2 Regularizer with the given regularization factors.
  """
    return L1L2(l1=l1, l2=l2)