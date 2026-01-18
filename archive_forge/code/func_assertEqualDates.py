import datetime
import logging
import os
import sys
import time
import threading
import types
import http.client
import cheroot.server
import cheroot.wsgi
from cheroot.test import webtest
def assertEqualDates(self, dt1, dt2, seconds=None):
    """Assert ``abs(dt1 - dt2)`` is within ``Y`` seconds."""
    if seconds is None:
        seconds = self.date_tolerance
    if dt1 > dt2:
        diff = dt1 - dt2
    else:
        diff = dt2 - dt1
    if not diff < datetime.timedelta(seconds=seconds):
        raise AssertionError('%r and %r are not within %r seconds.' % (dt1, dt2, seconds))