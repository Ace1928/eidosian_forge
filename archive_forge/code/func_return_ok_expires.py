import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def return_ok_expires(self, cookie, request):
    if cookie.is_expired(self._now):
        _debug('   cookie expired')
        return False
    return True