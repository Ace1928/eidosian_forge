import binascii
import codecs
import importlib
import marshal
import os
import re
import sys
import threading
import time
import types as python_types
import warnings
import weakref
import numpy as np
from tensorflow.python.keras.utils import tf_contextlib
from tensorflow.python.keras.utils import tf_inspect
from tensorflow.python.util import nest
from tensorflow.python.util import tf_decorator
def has_arg(fn, name, accept_all=False):
    """Checks if a callable accepts a given keyword argument.

  Args:
      fn: Callable to inspect.
      name: Check if `fn` can be called with `name` as a keyword argument.
      accept_all: What to return if there is no parameter called `name` but the
        function accepts a `**kwargs` argument.

  Returns:
      bool, whether `fn` accepts a `name` keyword argument.
  """
    arg_spec = tf_inspect.getfullargspec(fn)
    if accept_all and arg_spec.varkw is not None:
        return True
    return name in arg_spec.args or name in arg_spec.kwonlyargs