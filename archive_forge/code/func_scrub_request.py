from sentry_sdk.utils import (
from sentry_sdk._compat import string_types
from sentry_sdk._types import TYPE_CHECKING
def scrub_request(self, event):
    with capture_internal_exceptions():
        if 'request' in event:
            if 'headers' in event['request']:
                self.scrub_dict(event['request']['headers'])
            if 'cookies' in event['request']:
                self.scrub_dict(event['request']['cookies'])
            if 'data' in event['request']:
                self.scrub_dict(event['request']['data'])