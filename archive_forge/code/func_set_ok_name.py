import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def set_ok_name(self, cookie, request):
    if cookie.version == 0 and self.strict_ns_set_initial_dollar and cookie.name.startswith('$'):
        _debug("   illegal name (starts with '$'): '%s'", cookie.name)
        return False
    return True