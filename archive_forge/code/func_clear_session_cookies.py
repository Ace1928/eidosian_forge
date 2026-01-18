import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def clear_session_cookies(self):
    """Discard all session cookies.

        Note that the .save() method won't save session cookies anyway, unless
        you ask otherwise by passing a true ignore_discard argument.

        """
    self._cookies_lock.acquire()
    try:
        for cookie in self:
            if cookie.discard:
                self.clear(cookie.domain, cookie.path, cookie.name)
    finally:
        self._cookies_lock.release()