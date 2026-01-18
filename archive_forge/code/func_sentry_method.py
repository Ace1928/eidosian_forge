import functools
from typing import TYPE_CHECKING
from django import VERSION as DJANGO_VERSION
from django.core.cache import CacheHandler
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk._compat import text_type
@functools.wraps(original_method)
def sentry_method(*args, **kwargs):
    return _instrument_call(cache, method_name, original_method, args, kwargs)