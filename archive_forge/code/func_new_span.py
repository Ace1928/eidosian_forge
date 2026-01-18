import uuid
import random
from datetime import datetime, timedelta
import sentry_sdk
from sentry_sdk.consts import INSTRUMENTER
from sentry_sdk.utils import is_valid_sample_rate, logger, nanosecond_time
from sentry_sdk._compat import datetime_utcnow, utc_from_timestamp, PY2
from sentry_sdk.consts import SPANDATA
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.tracing_utils import (
from sentry_sdk.metrics import LocalAggregator
def new_span(self, **kwargs):
    return self.start_child(**kwargs)