from __future__ import unicode_literals
import datetime
import functools
from google.api_core import datetime_helpers
@functools.wraps(func)
def func_with_timeout(*args, **kwargs):
    """Wrapped function that adds timeout."""
    kwargs['timeout'] = next(timeouts)
    return func(*args, **kwargs)