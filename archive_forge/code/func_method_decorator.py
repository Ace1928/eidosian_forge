import warnings
from warnings import warn
import breezy
def method_decorator(callable):
    """This is the function python calls to perform the decoration."""

    def decorated_method(self, *args, **kwargs):
        """This is the decorated method."""
        from . import trace
        if callable.__name__ == '__init__':
            symbol = '{}.{}'.format(self.__class__.__module__, self.__class__.__name__)
        else:
            symbol = '{}.{}.{}'.format(self.__class__.__module__, self.__class__.__name__, callable.__name__)
        trace.mutter_callsite(4, 'Deprecated method called')
        warn(deprecation_version % symbol, DeprecationWarning, stacklevel=2)
        return callable(self, *args, **kwargs)
    _populate_decorated(callable, deprecation_version, 'method', decorated_method)
    return decorated_method