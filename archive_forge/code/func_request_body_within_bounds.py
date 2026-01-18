from __future__ import absolute_import
import json
from copy import deepcopy
from sentry_sdk.hub import Hub, _should_send_default_pii
from sentry_sdk.utils import AnnotatedValue
from sentry_sdk._compat import text_type, iteritems
from sentry_sdk._types import TYPE_CHECKING
def request_body_within_bounds(client, content_length):
    if client is None:
        return False
    bodies = client.options['max_request_body_size']
    return not (bodies == 'never' or (bodies == 'small' and content_length > 10 ** 3) or (bodies == 'medium' and content_length > 10 ** 4))