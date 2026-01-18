import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def getfile(object):
    """TFDecorator-aware replacement for inspect.getfile."""
    unwrapped_object = tf_decorator.unwrap(object)[1]
    if hasattr(unwrapped_object, 'f_globals') and '__file__' in unwrapped_object.f_globals:
        return unwrapped_object.f_globals['__file__']
    return _inspect.getfile(unwrapped_object)