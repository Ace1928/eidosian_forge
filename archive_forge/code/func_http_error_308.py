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
def http_error_308(self, url, fp, errcode, errmsg, headers, data=None):
    """Error 308 -- relocated, but turn POST into error."""
    if data is None:
        return self.http_error_301(url, fp, errcode, errmsg, headers, data)
    else:
        return self.http_error_default(url, fp, errcode, errmsg, headers)