import functools
import six
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
def get_func_code(func):
    """Returns func_code of passed callable, or None if not available."""
    _, func = tf_decorator.unwrap(func)
    if callable(func):
        if tf_inspect.isfunction(func) or tf_inspect.ismethod(func):
            return six.get_function_code(func)
        try:
            return six.get_function_code(func.__call__)
        except AttributeError:
            return None
    else:
        raise ValueError(f'Argument `func` must be a callable. Received func={func} (of type {type(func)})')