import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def return_ok(self, cookie, request):
    """
        If you override .return_ok(), be sure to call this method.  If it
        returns false, so should your subclass (assuming your subclass wants to
        be more strict about which cookies to return).

        """
    _debug(' - checking cookie %s=%s', cookie.name, cookie.value)
    for n in ('version', 'verifiability', 'secure', 'expires', 'port', 'domain'):
        fn_name = 'return_ok_' + n
        fn = getattr(self, fn_name)
        if not fn(cookie, request):
            return False
    return True