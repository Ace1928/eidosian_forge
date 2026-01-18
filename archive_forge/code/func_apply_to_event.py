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
@_disable_capture
def apply_to_event(self, event, hint, options=None):
    """Applies the information contained on the scope to the given event."""
    ty = event.get('type')
    is_transaction = ty == 'transaction'
    is_check_in = ty == 'check_in'
    attachments_to_send = hint.get('attachments') or []
    for attachment in self._attachments:
        if not is_transaction or attachment.add_to_transactions:
            attachments_to_send.append(attachment)
    hint['attachments'] = attachments_to_send
    self._apply_contexts_to_event(event, hint, options)
    if is_check_in:
        event['contexts'] = {'trace': event.setdefault('contexts', {}).get('trace', {})}
    if not is_check_in:
        self._apply_level_to_event(event, hint, options)
        self._apply_fingerprint_to_event(event, hint, options)
        self._apply_user_to_event(event, hint, options)
        self._apply_transaction_name_to_event(event, hint, options)
        self._apply_transaction_info_to_event(event, hint, options)
        self._apply_tags_to_event(event, hint, options)
        self._apply_extra_to_event(event, hint, options)
    if not is_transaction and (not is_check_in):
        self._apply_breadcrumbs_to_event(event, hint, options)

    def _drop(cause, ty):
        logger.info('%s (%s) dropped event', ty, cause)
        return None
    exc_info = hint.get('exc_info')
    if exc_info is not None:
        for error_processor in self._error_processors:
            new_event = error_processor(event, exc_info)
            if new_event is None:
                return _drop(error_processor, 'error processor')
            event = new_event
    if not is_check_in:
        for event_processor in chain(global_event_processors, self._event_processors):
            new_event = event
            with capture_internal_exceptions():
                new_event = event_processor(event, hint)
            if new_event is None:
                return _drop(event_processor, 'event processor')
            event = new_event
    return event