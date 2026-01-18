import warnings
from warnings import warn
import breezy
def function_decorator(callable):
    """This is the function python calls to perform the decoration."""

    def decorated_function(*args, **kwargs):
        """This is the decorated function."""
        from . import trace
        trace.mutter_callsite(4, 'Deprecated function called')
        warn(deprecation_string(callable, deprecation_version), DeprecationWarning, stacklevel=2)
        return callable(*args, **kwargs)
    _populate_decorated(callable, deprecation_version, 'function', decorated_function)
    return decorated_function