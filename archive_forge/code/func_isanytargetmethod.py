import collections
import functools
import inspect as _inspect
import six
from tensorflow.python.util import tf_decorator
def isanytargetmethod(object):
    """Checks if `object` or a TF Decorator wrapped target contains self or cls.

  This function could be used along with `tf_inspect.getfullargspec` to
  determine if the first argument of `object` argspec is self or cls. If the
  first argument is self or cls, it needs to be excluded from argspec when we
  compare the argspec to the input arguments and, if provided, the tf.function
  input_signature.

  Like `tf_inspect.getfullargspec` and python `inspect.getfullargspec`, it
  does not unwrap python decorators.

  Args:
    obj: An method, function, or functool.partial, possibly decorated by
    TFDecorator.

  Returns:
    A bool indicates if `object` or any target along the chain of TF decorators
    is a method.
  """
    decorators, target = tf_decorator.unwrap(object)
    for decorator in decorators:
        if _inspect.ismethod(decorator.decorated_target):
            return True
    while isinstance(target, functools.partial):
        target = target.func
    return callable(target) and (not _inspect.isfunction(target))