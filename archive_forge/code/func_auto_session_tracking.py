import os
import time
from threading import Thread, Lock
from contextlib import contextmanager
import sentry_sdk
from sentry_sdk.envelope import Envelope
from sentry_sdk.session import Session
from sentry_sdk._types import TYPE_CHECKING
from sentry_sdk.utils import format_timestamp
@contextmanager
def auto_session_tracking(hub=None, session_mode='application'):
    """Starts and stops a session automatically around a block."""
    if hub is None:
        hub = sentry_sdk.Hub.current
    should_track = is_auto_session_tracking_enabled(hub)
    if should_track:
        hub.start_session(session_mode=session_mode)
    try:
        yield
    finally:
        if should_track:
            hub.end_session()