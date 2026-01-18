import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def is_not_allowed(self, domain):
    if self._allowed_domains is None:
        return False
    for allowed_domain in self._allowed_domains:
        if user_domain_match(domain, allowed_domain):
            return False
    return True