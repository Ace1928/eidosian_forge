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
def func_dump(func):
    """Serializes a user defined function.

  Args:
      func: the function to serialize.

  Returns:
      A tuple `(code, defaults, closure)`.
  """
    if os.name == 'nt':
        raw_code = marshal.dumps(func.__code__).replace(b'\\', b'/')
        code = codecs.encode(raw_code, 'base64').decode('ascii')
    else:
        raw_code = marshal.dumps(func.__code__)
        code = codecs.encode(raw_code, 'base64').decode('ascii')
    defaults = func.__defaults__
    if func.__closure__:
        closure = tuple((c.cell_contents for c in func.__closure__))
    else:
        closure = None
    return (code, defaults, closure)