import collections
import functools
import inspect as _inspect
import six
from tensorflow.python.util import tf_decorator
def signature(obj, *, follow_wrapped=True):
    """TFDecorator-aware replacement for inspect.signature."""
    return _inspect.signature(tf_decorator.unwrap(obj)[1], follow_wrapped=follow_wrapped)