import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def isbuiltin(object):
    """TFDecorator-aware replacement for inspect.isbuiltin."""
    return _inspect.isbuiltin(tf_decorator.unwrap(object)[1])