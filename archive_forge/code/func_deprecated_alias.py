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
def deprecated_alias(deprecated_name, name, func_or_class, warn_once=True):
    """Deprecate a symbol in favor of a new name with identical semantics.

  This function is meant to be used when defining a backwards-compatibility
  alias for a symbol which has been moved. For example:

  module1.py:
  ```python
  class NewNameForClass: pass
  ```

  module2.py:
  ```python
  import module1

  DeprecatedNameForClass = deprecated_alias(
    deprecated_name='module2.DeprecatedNameForClass',
    name='module1.NewNameForClass',
    func_or_class=module1.NewNameForClass)
  ```

  This function works for classes and functions.

  For classes, it creates a new class which is functionally identical (it
  inherits from the original, and overrides its constructor), but which prints
  a deprecation warning when an instance is created. It also adds a deprecation
  notice to the class' docstring.

  For functions, it returns a function wrapped by `tf_decorator.make_decorator`.
  That function prints a warning when used, and has a deprecation notice in its
  docstring. This is more or less equivalent (the deprecation warning has
  slightly different text) to writing:

  ```python
  @deprecated
  def deprecated_alias(original_args):
    real_function(original_args)
  ```

  Args:
    deprecated_name: The name of the symbol that is being deprecated, to be used
      in the warning message. This should be its fully qualified name to avoid
      confusion.
    name: The name of the symbol that is to be used instead of the deprecated
      name. This should be a fully qualified name to avoid confusion.
    func_or_class: The (non-deprecated) class or function for which a deprecated
      alias should be created.
    warn_once: If True (the default), only print a deprecation warning the first
      time this function is used, or the class is instantiated.

  Returns:
    A wrapped version of `func_or_class` which prints a deprecation warning on
    use and has a modified docstring.
  """
    if tf_inspect.isclass(func_or_class):

        class _NewClass(func_or_class):
            __doc__ = decorator_utils.add_notice_to_docstring(func_or_class.__doc__, 'Please use %s instead.' % name, 'DEPRECATED CLASS', '(deprecated)', ['THIS CLASS IS DEPRECATED. It will be removed in a future version. '], notice_type='Deprecated')
            __name__ = func_or_class.__name__
            __module__ = _call_location(outer=True)

            @_wrap_decorator(func_or_class.__init__, 'deprecated_alias')
            def __init__(self, *args, **kwargs):
                if hasattr(_NewClass.__init__, '__func__'):
                    _NewClass.__init__.__func__.__doc__ = func_or_class.__init__.__doc__
                else:
                    _NewClass.__init__.__doc__ = func_or_class.__init__.__doc__
                if _PRINT_DEPRECATION_WARNINGS:
                    if _NewClass.__init__ not in _PRINTED_WARNING:
                        if warn_once:
                            _PRINTED_WARNING[_NewClass.__init__] = True
                        _log_deprecation('From %s: The name %s is deprecated. Please use %s instead.\n', _call_location(), deprecated_name, name)
                super(_NewClass, self).__init__(*args, **kwargs)
        return _NewClass
    else:
        decorator_utils.validate_callable(func_or_class, 'deprecated')

        @functools.wraps(func_or_class)
        def new_func(*args, **kwargs):
            if _PRINT_DEPRECATION_WARNINGS:
                if new_func not in _PRINTED_WARNING:
                    if warn_once:
                        _PRINTED_WARNING[new_func] = True
                    _log_deprecation('From %s: The name %s is deprecated. Please use %s instead.\n', _call_location(), deprecated_name, name)
            return func_or_class(*args, **kwargs)
        return tf_decorator.make_decorator(func_or_class, new_func, 'deprecated', _add_deprecated_function_notice_to_docstring(func_or_class.__doc__, None, 'Please use %s instead.' % name))