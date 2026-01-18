import abc
import collections
import contextlib
import re
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import util as distribution_util
from tensorflow.python.util import object_identity
def inverse(self, y, name='inverse'):
    """Returns the inverse `Bijector` evaluation, i.e., X = g^{-1}(Y).

    Args:
      y: `Tensor`. The input to the "inverse" evaluation.
      name: The name to give this op.

    Returns:
      `Tensor`, if this bijector is injective.
        If not injective, returns the k-tuple containing the unique
        `k` points `(x1, ..., xk)` such that `g(xi) = y`.

    Raises:
      TypeError: if `self.dtype` is specified and `y.dtype` is not
        `self.dtype`.
      NotImplementedError: if `_inverse` is not implemented.
    """
    return self._call_inverse(y, name)