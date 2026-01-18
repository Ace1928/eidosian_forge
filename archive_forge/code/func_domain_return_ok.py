import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def domain_return_ok(self, domain, request):
    req_host, erhn = eff_request_host(request)
    if not req_host.startswith('.'):
        req_host = '.' + req_host
    if not erhn.startswith('.'):
        erhn = '.' + erhn
    if domain and (not domain.startswith('.')):
        dotdomain = '.' + domain
    else:
        dotdomain = domain
    if not (req_host.endswith(dotdomain) or erhn.endswith(dotdomain)):
        return False
    if self.is_blocked(domain):
        _debug('   domain %s is in user block-list', domain)
        return False
    if self.is_not_allowed(domain):
        _debug('   domain %s is not in user allow-list', domain)
        return False
    return True