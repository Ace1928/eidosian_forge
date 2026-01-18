from sentry_sdk.utils import (
from sentry_sdk._compat import string_types
from sentry_sdk._types import TYPE_CHECKING
def scrub_breadcrumbs(self, event):
    with capture_internal_exceptions():
        if 'breadcrumbs' in event:
            if 'values' in event['breadcrumbs']:
                for value in event['breadcrumbs']['values']:
                    if 'data' in value:
                        self.scrub_dict(value['data'])