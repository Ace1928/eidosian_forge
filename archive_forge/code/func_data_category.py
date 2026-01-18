import io
import json
import mimetypes
from sentry_sdk._compat import text_type, PY2
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.session import Session
from sentry_sdk.utils import json_dumps, capture_internal_exceptions
@property
def data_category(self):
    ty = self.headers.get('type')
    if ty == 'session':
        return 'session'
    elif ty == 'attachment':
        return 'attachment'
    elif ty == 'transaction':
        return 'transaction'
    elif ty == 'event':
        return 'error'
    elif ty == 'client_report':
        return 'internal'
    elif ty == 'profile':
        return 'profile'
    elif ty == 'statsd':
        return 'statsd'
    elif ty == 'check_in':
        return 'monitor'
    else:
        return 'default'