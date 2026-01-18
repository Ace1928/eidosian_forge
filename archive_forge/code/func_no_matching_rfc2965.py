import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def no_matching_rfc2965(ns_cookie, lookup=lookup):
    key = (ns_cookie.domain, ns_cookie.path, ns_cookie.name)
    return key not in lookup