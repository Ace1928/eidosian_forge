from copy import copy
from collections import deque
from itertools import chain
import os
import sys
import uuid
from sentry_sdk.attachments import Attachment
from sentry_sdk._compat import datetime_utcnow
from sentry_sdk.consts import FALSE_VALUES, INSTRUMENTER
from sentry_sdk._functools import wraps
from sentry_sdk.profiler import Profile
from sentry_sdk.session import Session
from sentry_sdk.tracing_utils import (
from sentry_sdk.tracing import (
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import (
@span.setter
def span(self, span):
    self._span = span
    if isinstance(span, Transaction):
        transaction = span
        if transaction.name:
            self._transaction = transaction.name
            if transaction.source:
                self._transaction_info['source'] = transaction.source