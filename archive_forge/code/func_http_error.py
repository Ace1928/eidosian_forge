import urllib.request
import base64
import bisect
import email
import hashlib
import http.client
import io
import os
import posixpath
import re
import socket
import string
import sys
import time
import tempfile
import contextlib
import warnings
from urllib.error import URLError, HTTPError, ContentTooShortError
from urllib.parse import (
from urllib.response import addinfourl, addclosehook
def http_error(self, url, fp, errcode, errmsg, headers, data=None):
    """Handle http errors.

        Derived class can override this, or provide specific handlers
        named http_error_DDD where DDD is the 3-digit error code."""
    name = 'http_error_%d' % errcode
    if hasattr(self, name):
        method = getattr(self, name)
        if data is None:
            result = method(url, fp, errcode, errmsg, headers)
        else:
            result = method(url, fp, errcode, errmsg, headers, data)
        if result:
            return result
    return self.http_error_default(url, fp, errcode, errmsg, headers)