import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def isgenerator(object):
    """TFDecorator-aware replacement for inspect.isgenerator."""
    return _inspect.isgenerator(tf_decorator.unwrap(object)[1])