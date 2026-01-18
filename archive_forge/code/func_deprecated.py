import collections
import functools
import inspect
import re
from tensorflow.python.framework import strict_mode
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import decorator_utils
from tensorflow.python.util import is_in_graph_mode
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util import tf_decorator
from tensorflow.python.util import tf_inspect
from tensorflow.tools.docs import doc_controls
def deprecated(date, instructions, warn_once=True):
    """Decorator for marking functions or methods deprecated.

  This decorator logs a deprecation warning whenever the decorated function is
  called. It has the following format:

    <function> (from <module>) is deprecated and will be removed after <date>.
    Instructions for updating:
    <instructions>

  If `date` is None, 'after <date>' is replaced with 'in a future version'.
  <function> will include the class name if it is a method.

  It also edits the docstring of the function: ' (deprecated)' is appended
  to the first line of the docstring and a deprecation notice is prepended
  to the rest of the docstring.

  Args:
    date: String or None. The date the function is scheduled to be removed. Must
      be ISO 8601 (YYYY-MM-DD), or None.
    instructions: String. Instructions on how to update code using the
      deprecated function.
    warn_once: Boolean. Set to `True` to warn only the first time the decorated
      function is called. Otherwise, every call will log a warning.

  Returns:
    Decorated function or method.

  Raises:
    ValueError: If date is not None or in ISO 8601 format, or instructions are
      empty.
  """
    _validate_deprecation_args(date, instructions)

    def deprecated_wrapper(func_or_class):
        """Deprecation wrapper."""
        if isinstance(func_or_class, type):
            cls = func_or_class
            if cls.__new__ is object.__new__:
                func = cls.__init__
                constructor_name = '__init__'
                decorators, _ = tf_decorator.unwrap(func)
                for decorator in decorators:
                    if decorator.decorator_name == 'deprecated':
                        return cls
            else:
                func = cls.__new__
                constructor_name = '__new__'
        else:
            cls = None
            constructor_name = None
            func = func_or_class
        decorator_utils.validate_callable(func, 'deprecated')

        @_wrap_decorator(func, 'deprecated')
        def new_func(*args, **kwargs):
            if _PRINT_DEPRECATION_WARNINGS:
                if func not in _PRINTED_WARNING and cls not in _PRINTED_WARNING:
                    if warn_once:
                        _PRINTED_WARNING[func] = True
                        if cls:
                            _PRINTED_WARNING[cls] = True
                    _log_deprecation('From %s: %s (from %s) is deprecated and will be removed %s.\nInstructions for updating:\n%s', _call_location(), decorator_utils.get_qualified_name(func), func_or_class.__module__, 'in a future version' if date is None else 'after %s' % date, instructions)
            return func(*args, **kwargs)
        doc_controls.set_deprecated(new_func)
        new_func = tf_decorator.make_decorator(func, new_func, 'deprecated', _add_deprecated_function_notice_to_docstring(func.__doc__, date, instructions))
        new_func.__signature__ = inspect.signature(func)
        if cls is None:
            return new_func
        else:
            setattr(cls, constructor_name, new_func)
            cls.__doc__ = _add_deprecated_function_notice_to_docstring(cls.__doc__, date, instructions)
            return cls
    return deprecated_wrapper