import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def getmembers(object, predicate=None):
    """TFDecorator-aware replacement for inspect.getmembers."""
    return _inspect.getmembers(object, predicate)