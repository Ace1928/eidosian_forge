import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def return_ok_verifiability(self, cookie, request):
    if request.unverifiable and is_third_party(request):
        if cookie.version > 0 and self.strict_rfc2965_unverifiable:
            _debug('   third-party RFC 2965 cookie during unverifiable transaction')
            return False
        elif cookie.version == 0 and self.strict_ns_unverifiable:
            _debug('   third-party Netscape cookie during unverifiable transaction')
            return False
    return True