import functools
from typing import TYPE_CHECKING
from django import VERSION as DJANGO_VERSION
from django.core.cache import CacheHandler
from sentry_sdk import Hub
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk._compat import text_type
@functools.wraps(original_create_connection)
def sentry_create_connection(self, alias):
    cache = original_create_connection(self, alias)
    integration = Hub.current.get_integration(DjangoIntegration)
    if integration and integration.cache_spans:
        _patch_cache(cache)
    return cache