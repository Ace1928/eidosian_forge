import copy
import time
import calendar
from ._internal_utils import to_native_string
from .compat import cookielib, urlparse, urlunparse, Morsel, MutableMapping
def multiple_domains(self):
    """Returns True if there are multiple domains in the jar.
        Returns False otherwise.

        :rtype: bool
        """
    domains = []
    for cookie in iter(self):
        if cookie.domain is not None and cookie.domain in domains:
            return True
        domains.append(cookie.domain)
    return False