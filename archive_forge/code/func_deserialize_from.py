import io
import json
import mimetypes
from sentry_sdk._compat import text_type, PY2
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.session import Session
from sentry_sdk.utils import json_dumps, capture_internal_exceptions
@classmethod
def deserialize_from(cls, f):
    line = f.readline().rstrip()
    if not line:
        return None
    headers = parse_json(line)
    length = headers.get('length')
    if length is not None:
        payload = f.read(length)
        f.readline()
    else:
        payload = f.readline().rstrip(b'\n')
    if headers.get('type') in ('event', 'transaction', 'metric_buckets'):
        rv = cls(headers=headers, payload=PayloadRef(json=parse_json(payload)))
    else:
        rv = cls(headers=headers, payload=payload)
    return rv