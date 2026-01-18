import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def isgeneratorfunction(object):
    """TFDecorator-aware replacement for inspect.isgeneratorfunction."""
    return _inspect.isgeneratorfunction(tf_decorator.unwrap(object)[1])