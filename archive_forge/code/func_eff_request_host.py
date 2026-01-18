import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def eff_request_host(request):
    """Return a tuple (request-host, effective request-host name).

    As defined by RFC 2965, except both are lowercased.

    """
    erhn = req_host = request_host(request)
    if req_host.find('.') == -1 and (not IPV4_RE.search(req_host)):
        erhn = req_host + '.local'
    return (req_host, erhn)