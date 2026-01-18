import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def clear_expired_cookies(self):
    """Discard all expired cookies.

        You probably don't need to call this method: expired cookies are never
        sent back to the server (provided you're using DefaultCookiePolicy),
        this method is called by CookieJar itself every so often, and the
        .save() method won't save expired cookies anyway (unless you ask
        otherwise by passing a true ignore_expires argument).

        """
    self._cookies_lock.acquire()
    try:
        now = time.time()
        for cookie in self:
            if cookie.is_expired(now):
                self.clear(cookie.domain, cookie.path, cookie.name)
    finally:
        self._cookies_lock.release()