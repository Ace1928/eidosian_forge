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
def populate_from_transaction(cls, transaction):
    """
        Populate fresh baggage entry with sentry_items and make it immutable
        if this is the head SDK which originates traces.
        """
    hub = transaction.hub or sentry_sdk.Hub.current
    client = hub.client
    sentry_items = {}
    if not client:
        return Baggage(sentry_items)
    options = client.options or {}
    user = hub.scope and hub.scope._user or {}
    sentry_items['trace_id'] = transaction.trace_id
    if options.get('environment'):
        sentry_items['environment'] = options['environment']
    if options.get('release'):
        sentry_items['release'] = options['release']
    if options.get('dsn'):
        sentry_items['public_key'] = Dsn(options['dsn']).public_key
    if transaction.name and transaction.source not in LOW_QUALITY_TRANSACTION_SOURCES:
        sentry_items['transaction'] = transaction.name
    if user.get('segment'):
        sentry_items['user_segment'] = user['segment']
    if transaction.sample_rate is not None:
        sentry_items['sample_rate'] = str(transaction.sample_rate)
    if transaction.sampled is not None:
        sentry_items['sampled'] = 'true' if transaction.sampled else 'false'
    if transaction._baggage and transaction._baggage.sentry_items:
        sentry_items.update(transaction._baggage.sentry_items)
    return Baggage(sentry_items, mutable=False)