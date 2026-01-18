import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def return_ok_secure(self, cookie, request):
    if cookie.secure and request.type not in self.secure_protocols:
        _debug('   secure cookie with non-secure request')
        return False
    return True