import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def getargspec(obj):
    """TFDecorator-aware replacement for `inspect.getargspec`.

  Note: `getfullargspec` is recommended as the python 2/3 compatible
  replacement for this function.

  Args:
    obj: A function, partial function, or callable object, possibly decorated.

  Returns:
    The `ArgSpec` that describes the signature of the outermost decorator that
    changes the callable's signature, or the `ArgSpec` that describes
    the object if not decorated.

  Raises:
    ValueError: When callable's signature can not be expressed with
      ArgSpec.
    TypeError: For objects of unsupported types.
  """
    if isinstance(obj, functools.partial):
        return _get_argspec_for_partial(obj)
    decorators, target = tf_decorator.unwrap(obj)
    spec = next((d.decorator_argspec for d in decorators if d.decorator_argspec is not None), None)
    if spec:
        return spec
    try:
        return _getargspec(target)
    except TypeError:
        pass
    if isinstance(target, type):
        try:
            return _getargspec(target.__init__)
        except TypeError:
            pass
        try:
            return _getargspec(target.__new__)
        except TypeError:
            pass
    return _getargspec(type(target).__call__)