import os
import copy
import datetime
import re
import time
import urllib.parse, urllib.request
import threading as _threading
import http.client  # only for the default HTTP port
from calendar import timegm
def set_blocked_domains(self, blocked_domains):
    """Set the sequence of blocked domains."""
    self._blocked_domains = tuple(blocked_domains)