import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def return_ok_port(self, cookie, request):
    if cookie.port:
        req_port = request_port(request)
        if req_port is None:
            req_port = '80'
        for p in cookie.port.split(','):
            if p == req_port:
                break
        else:
            _debug('   request port %s does not match cookie port %s', req_port, cookie.port)
            return False
    return True