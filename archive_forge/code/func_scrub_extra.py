from sentry_sdk.utils import (
from sentry_sdk._compat import string_types
from sentry_sdk._types import TYPE_CHECKING
def scrub_extra(self, event):
    with capture_internal_exceptions():
        if 'extra' in event:
            self.scrub_dict(event['extra'])