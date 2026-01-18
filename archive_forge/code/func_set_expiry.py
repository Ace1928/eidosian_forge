import logging
import string
from datetime import datetime, timedelta
from django.conf import settings
from django.core import signing
from django.utils import timezone
from django.utils.crypto import get_random_string
from django.utils.module_loading import import_string
def set_expiry(self, value):
    """
        Set a custom expiration for the session. ``value`` can be an integer,
        a Python ``datetime`` or ``timedelta`` object or ``None``.

        If ``value`` is an integer, the session will expire after that many
        seconds of inactivity. If set to ``0`` then the session will expire on
        browser close.

        If ``value`` is a ``datetime`` or ``timedelta`` object, the session
        will expire at that specific future time.

        If ``value`` is ``None``, the session uses the global session expiry
        policy.
        """
    if value is None:
        try:
            del self['_session_expiry']
        except KeyError:
            pass
        return
    if isinstance(value, timedelta):
        value = timezone.now() + value
    if isinstance(value, datetime):
        value = value.isoformat()
    self['_session_expiry'] = value