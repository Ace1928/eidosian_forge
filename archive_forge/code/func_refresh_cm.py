import sys
import warnings
from collections import deque
from functools import wraps
def refresh_cm(self):
    """Returns the context manager used to actually wrap the call to the
        decorated function.

        The default implementation just returns *self*.

        Overriding this method allows otherwise one-shot context managers
        like _GeneratorContextManager to support use as decorators via
        implicit recreation.

        DEPRECATED: refresh_cm was never added to the standard library's
                    ContextDecorator API
        """
    warnings.warn('refresh_cm was never added to the standard library', DeprecationWarning)
    return self._recreate_cm()