import collections
import functools
import inspect as _inspect
from tensorflow.python.util import tf_decorator
def getcallargs(*func_and_positional, **named):
    """TFDecorator-aware replacement for inspect.getcallargs.

  Args:
    *func_and_positional: A callable, possibly decorated, followed by any
      positional arguments that would be passed to `func`.
    **named: The named argument dictionary that would be passed to `func`.

  Returns:
    A dictionary mapping `func`'s named arguments to the values they would
    receive if `func(*positional, **named)` were called.

  `getcallargs` will use the argspec from the outermost decorator that provides
  it. If no attached decorators modify argspec, the final unwrapped target's
  argspec will be used.
  """
    func = func_and_positional[0]
    positional = func_and_positional[1:]
    argspec = getfullargspec(func)
    call_args = named.copy()
    this = getattr(func, 'im_self', None) or getattr(func, '__self__', None)
    if ismethod(func) and this:
        positional = (this,) + positional
    remaining_positionals = [arg for arg in argspec.args if arg not in call_args]
    call_args.update(dict(zip(remaining_positionals, positional)))
    default_count = 0 if not argspec.defaults else len(argspec.defaults)
    if default_count:
        for arg, value in zip(argspec.args[-default_count:], argspec.defaults):
            if arg not in call_args:
                call_args[arg] = value
    if argspec.kwonlydefaults is not None:
        for k, v in argspec.kwonlydefaults.items():
            if k not in call_args:
                call_args[k] = v
    return call_args