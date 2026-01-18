import collections
import contextlib
import functools
import itertools
import threading
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.eager import context
from tensorflow.python.framework import config
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import test_util
from tensorflow.python.keras import backend
from tensorflow.python.keras import layers
from tensorflow.python.keras import models
from tensorflow.python.keras.engine import base_layer_utils
from tensorflow.python.keras.optimizer_v2 import adadelta as adadelta_v2
from tensorflow.python.keras.optimizer_v2 import adagrad as adagrad_v2
from tensorflow.python.keras.optimizer_v2 import adam as adam_v2
from tensorflow.python.keras.optimizer_v2 import adamax as adamax_v2
from tensorflow.python.keras.optimizer_v2 import gradient_descent as gradient_descent_v2
from tensorflow.python.keras.optimizer_v2 import nadam as nadam_v2
from tensorflow.python.keras.optimizer_v2 import rmsprop as rmsprop_v2
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.util import tf_decorator
def run_without_tensor_float_32(description):
    """Execute test with TensorFloat-32 disabled.

  While almost every real-world deep learning model runs fine with
  TensorFloat-32, many tests use assertAllClose or similar methods.
  TensorFloat-32 matmuls typically will cause such methods to fail with the
  default tolerances.

  Args:
    description: A description used for documentation purposes, describing why
      the test requires TensorFloat-32 to be disabled.

  Returns:
    Decorator which runs a test with TensorFloat-32 disabled.
  """

    def decorator(f):

        @functools.wraps(f)
        def decorated(self, *args, **kwargs):
            allowed = config.tensor_float_32_execution_enabled()
            try:
                config.enable_tensor_float_32_execution(False)
                f(self, *args, **kwargs)
            finally:
                config.enable_tensor_float_32_execution(allowed)
        return decorated
    return decorator