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
def for_all_test_methods(decorator, *args, **kwargs):
    """Generate class-level decorator from given method-level decorator.

  It is expected for the given decorator to take some arguments and return
  a method that is then called on the test method to produce a decorated
  method.

  Args:
    decorator: The decorator to apply.
    *args: Positional arguments
    **kwargs: Keyword arguments
  Returns: Function that will decorate a given classes test methods with the
    decorator.
  """

    def all_test_methods_impl(cls):
        """Apply decorator to all test methods in class."""
        for name in dir(cls):
            value = getattr(cls, name)
            if callable(value) and name.startswith('test') and (name != 'test_session'):
                setattr(cls, name, decorator(*args, **kwargs)(value))
        return cls
    return all_test_methods_impl