import logging
import string
from datetime import datetime, timedelta
from django.conf import settings
from django.core import signing
from django.utils import timezone
from django.utils.crypto import get_random_string
from django.utils.module_loading import import_string
def get_expiry_date(self, **kwargs):
    """Get session the expiry date (as a datetime object).

        Optionally, this function accepts `modification` and `expiry` keyword
        arguments specifying the modification and expiry of the session.
        """
    try:
        modification = kwargs['modification']
    except KeyError:
        modification = timezone.now()
    try:
        expiry = kwargs['expiry']
    except KeyError:
        expiry = self.get('_session_expiry')
    if isinstance(expiry, datetime):
        return expiry
    elif isinstance(expiry, str):
        return datetime.fromisoformat(expiry)
    expiry = expiry or self.get_session_cookie_age()
    return modification + timedelta(seconds=expiry)