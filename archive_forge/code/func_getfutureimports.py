import builtins
import inspect
import itertools
import linecache
import sys
import threading
import types
from tensorflow.python.util import tf_inspect
def getfutureimports(entity):
    """Detects what future imports are necessary to safely execute entity source.

  Args:
    entity: Any object

  Returns:
    A tuple of future strings
  """
    if not (tf_inspect.isfunction(entity) or tf_inspect.ismethod(entity)):
        return tuple()
    return tuple(sorted((name for name, value in entity.__globals__.items() if getattr(value, '__module__', None) == '__future__')))