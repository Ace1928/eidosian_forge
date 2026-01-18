import functools
import typing
from base64 import urlsafe_b64decode
from base64 import urlsafe_b64encode
from enum import Enum
def required_virtual_authenticator(func):
    """A decorator to ensure that the function is called with a virtual
    authenticator."""

    @functools.wraps(func)
    @required_chromium_based_browser
    def wrapper(self, *args, **kwargs):
        if not self.virtual_authenticator_id:
            raise ValueError('This function requires a virtual authenticator to be set.')
        return func(self, *args, **kwargs)
    return wrapper