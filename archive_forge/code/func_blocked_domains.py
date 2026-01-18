import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def blocked_domains(self):
    """Return the sequence of blocked domains (as a tuple)."""
    return self._blocked_domains