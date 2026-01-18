import contextlib
import os
import re
import sys
import sentry_sdk
from sentry_sdk.consts import OP, SPANDATA
from sentry_sdk.utils import (
from sentry_sdk._compat import PY2, duration_in_milliseconds, iteritems
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.tracing import LOW_QUALITY_TRANSACTION_SOURCES
@classmethod
def from_incoming_header(cls, header):
    """
        freeze if incoming header already has sentry baggage
        """
    sentry_items = {}
    third_party_items = ''
    mutable = True
    if header:
        for item in header.split(','):
            if '=' not in item:
                continue
            with capture_internal_exceptions():
                item = item.strip()
                key, val = item.split('=')
                if Baggage.SENTRY_PREFIX_REGEX.match(key):
                    baggage_key = unquote(key.split('-')[1])
                    sentry_items[baggage_key] = unquote(val)
                    mutable = False
                else:
                    third_party_items += (',' if third_party_items else '') + item
    return Baggage(sentry_items, third_party_items, mutable)