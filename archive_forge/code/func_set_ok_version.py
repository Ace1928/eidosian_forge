import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def set_ok_version(self, cookie, request):
    if cookie.version is None:
        _debug('   Set-Cookie2 without version attribute (%s=%s)', cookie.name, cookie.value)
        return False
    if cookie.version > 0 and (not self.rfc2965):
        _debug('   RFC 2965 cookies are switched off')
        return False
    elif cookie.version == 0 and (not self.netscape):
        _debug('   Netscape cookies are switched off')
        return False
    return True