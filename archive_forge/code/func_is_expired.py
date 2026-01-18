import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def is_expired(self, now=None):
    if now is None:
        now = time.time()
    if self.expires is not None and self.expires <= now:
        return True
    return False