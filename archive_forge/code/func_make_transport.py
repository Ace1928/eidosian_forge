from __future__ import print_function
import io
import gzip
import time
from datetime import timedelta
from collections import defaultdict
import urllib3
import certifi
from sentry_sdk.utils import Dsn, logger, capture_internal_exceptions, json_dumps
from sentry_sdk.worker import BackgroundWorker
from sentry_sdk.envelope import Envelope, Item, PayloadRef
from sentry_sdk._compat import datetime_utcnow
from sentry_sdk._types import TYPE_CHECKING
def make_transport(options):
    ref_transport = options['transport']
    if ref_transport is None:
        transport_cls = HttpTransport
    elif isinstance(ref_transport, Transport):
        return ref_transport
    elif isinstance(ref_transport, type) and issubclass(ref_transport, Transport):
        transport_cls = ref_transport
    elif callable(ref_transport):
        return _FunctionTransport(ref_transport)
    if options['dsn']:
        return transport_cls(options)
    return None