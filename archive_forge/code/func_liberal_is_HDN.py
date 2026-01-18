import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def liberal_is_HDN(text):
    """Return True if text is a sort-of-like a host domain name.

    For accepting/blocking domains.

    """
    if IPV4_RE.search(text):
        return False
    return True