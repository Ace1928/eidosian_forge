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
@functools.wraps(func)
def new_func(*args, **kwargs):
    """Deprecation wrapper."""
    if _PRINT_DEPRECATION_WARNINGS:
        named_args = tf_inspect.getcallargs(func, *args, **kwargs)
        for arg_name, arg_value in deprecated_kwargs.items():
            if arg_name in named_args and _safe_eq(named_args[arg_name], arg_value):
                if (func, arg_name) not in _PRINTED_WARNING:
                    if warn_once:
                        _PRINTED_WARNING[func, arg_name] = True
                    _log_deprecation('From %s: calling %s (from %s) with %s=%s is deprecated and will be removed %s.\nInstructions for updating:\n%s', _call_location(), decorator_utils.get_qualified_name(func), func.__module__, arg_name, arg_value, 'in a future version' if date is None else 'after %s' % date, instructions)
    return func(*args, **kwargs)