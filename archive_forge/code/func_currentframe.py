import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def currentframe():
    """TFDecorator-aware replacement for inspect.currentframe."""
    return _inspect.stack()[1][0]