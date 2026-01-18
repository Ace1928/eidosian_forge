import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def set_ok_port(self, cookie, request):
    if cookie.port_specified:
        req_port = request_port(request)
        if req_port is None:
            req_port = '80'
        else:
            req_port = str(req_port)
        for p in cookie.port.split(','):
            try:
                int(p)
            except ValueError:
                _debug('   bad port %s (not numeric)', p)
                return False
            if p == req_port:
                break
        else:
            _debug('   request port (%s) not found in %s', req_port, cookie.port)
            return False
    return True