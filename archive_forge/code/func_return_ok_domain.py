import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def return_ok_domain(self, cookie, request):
    req_host, erhn = eff_request_host(request)
    domain = cookie.domain
    if domain and (not domain.startswith('.')):
        dotdomain = '.' + domain
    else:
        dotdomain = domain
    if cookie.version == 0 and self.strict_ns_domain & self.DomainStrictNonDomain and (not cookie.domain_specified) and (domain != erhn):
        _debug('   cookie with unspecified domain does not string-compare equal to request domain')
        return False
    if cookie.version > 0 and (not domain_match(erhn, domain)):
        _debug('   effective request-host name %s does not domain-match RFC 2965 cookie domain %s', erhn, domain)
        return False
    if cookie.version == 0 and (not ('.' + erhn).endswith(dotdomain)):
        _debug('   request-host %s does not match Netscape cookie domain %s', req_host, domain)
        return False
    return True