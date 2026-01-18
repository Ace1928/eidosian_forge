import copy
import time
import calendar
from ._internal_utils import to_native_string
from .compat import cookielib, urlparse, urlunparse, Morsel, MutableMapping
def list_domains(self):
    """Utility method to list all the domains in the jar."""
    domains = []
    for cookie in iter(self):
        if cookie.domain not in domains:
            domains.append(cookie.domain)
    return domains