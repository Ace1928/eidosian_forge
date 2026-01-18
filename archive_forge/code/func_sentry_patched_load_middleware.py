from django import VERSION as DJANGO_VERSION
from sentry_sdk import Hub
from sentry_sdk._functools import wraps
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.consts import OP
from sentry_sdk.utils import (
def sentry_patched_load_middleware(*args, **kwargs):
    _import_string_should_wrap_middleware.set(True)
    try:
        return old_load_middleware(*args, **kwargs)
    finally:
        _import_string_should_wrap_middleware.set(False)