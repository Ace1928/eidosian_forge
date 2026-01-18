import builtins
import inspect
import itertools
import linecache
import sys
import threading
import types
from tensorflow.python.util import tf_inspect
def isconstructor(cls):
    """Returns True if the argument is an object constructor.

  In general, any object of type class is a constructor, with the exception
  of classes created using a callable metaclass.
  See below for why a callable metaclass is not a trivial combination:
  https://docs.python.org/2.7/reference/datamodel.html#customizing-class-creation

  Args:
    cls: Any

  Returns:
    Bool
  """
    return inspect.isclass(cls) and (not (issubclass(cls.__class__, type) and hasattr(cls.__class__, '__call__') and (cls.__class__.__call__ is not type.__call__)))