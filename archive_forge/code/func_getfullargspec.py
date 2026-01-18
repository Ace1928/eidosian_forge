import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def getfullargspec(obj):
    """TFDecorator-aware replacement for `inspect.getfullargspec`.

  This wrapper emulates `inspect.getfullargspec` in[^)]* Python2.

  Args:
    obj: A callable, possibly decorated.

  Returns:
    The `FullArgSpec` that describes the signature of
    the outermost decorator that changes the callable's signature. If the
    callable is not decorated, `inspect.getfullargspec()` will be called
    directly on the callable.
  """
    decorators, target = tf_decorator.unwrap(obj)
    for d in decorators:
        if d.decorator_argspec is not None:
            return _convert_maybe_argspec_to_fullargspec(d.decorator_argspec)
    return _getfullargspec(target)