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
@tf_contextlib.contextmanager
def saved_model_format_scope(value, **kwargs):
    """Provides a scope within which the savde model format to test is `value`.

  The saved model format gets restored to its original value upon exiting the
  scope.

  Args:
     value: saved model format value
     **kwargs: optional kwargs to pass to the save function.

  Yields:
    The provided value.
  """
    previous_format = _thread_local_data.saved_model_format
    previous_kwargs = _thread_local_data.save_kwargs
    try:
        _thread_local_data.saved_model_format = value
        _thread_local_data.save_kwargs = kwargs
        yield
    finally:
        _thread_local_data.saved_model_format = previous_format
        _thread_local_data.save_kwargs = previous_kwargs