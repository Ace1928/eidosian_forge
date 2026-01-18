from __future__ import absolute_import
import re
from sentry_sdk._types import TYPE_CHECKING
from django import VERSION as DJANGO_VERSION
def get_regex(resolver_or_pattern):
    """Utility method for django's deprecated resolver.regex"""
    try:
        regex = resolver_or_pattern.regex
    except AttributeError:
        regex = resolver_or_pattern.pattern.regex
    return regex