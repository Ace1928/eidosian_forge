import builtins
import inspect
import itertools
import linecache
import sys
import threading
import types
from tensorflow.python.util import tf_inspect
def getnamespace(f):
    """Returns the complete namespace of a function.

  Namespace is defined here as the mapping of all non-local variables to values.
  This includes the globals and the closure variables. Note that this captures
  the entire globals collection of the function, and may contain extra symbols
  that it does not actually use.

  Args:
    f: User defined function.

  Returns:
    A dict mapping symbol names to values.
  """
    namespace = dict(f.__globals__)
    closure = f.__closure__
    freevars = f.__code__.co_freevars
    if freevars and closure:
        for name, cell in zip(freevars, closure):
            try:
                namespace[name] = cell.cell_contents
            except ValueError:
                pass
    return namespace