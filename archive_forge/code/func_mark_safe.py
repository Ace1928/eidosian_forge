from functools import wraps
from django.utils.functional import keep_lazy
@keep_lazy(SafeString)
def mark_safe(s):
    """
    Explicitly mark a string as safe for (HTML) output purposes. The returned
    object can be used everywhere a string is appropriate.

    If used on a method as a decorator, mark the returned data as safe.

    Can be called multiple times on a single string.
    """
    if hasattr(s, '__html__'):
        return s
    if callable(s):
        return _safety_decorator(mark_safe, s)
    return SafeString(s)