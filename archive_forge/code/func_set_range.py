import cgi
import email.utils as email_utils
import http.client as http_client
import os
from io import BytesIO
from ... import errors, osutils
def set_range(self, start, size):
    """Change the range mapping"""
    self._start = start
    self._size = size
    self._pos = self._start