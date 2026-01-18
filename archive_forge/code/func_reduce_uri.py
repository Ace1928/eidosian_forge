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
def reduce_uri(self, uri, default_port=True):
    """Accept authority or URI and extract only the authority and path."""
    parts = urlsplit(uri)
    if parts[1]:
        scheme = parts[0]
        authority = parts[1]
        path = parts[2] or '/'
    else:
        scheme = None
        authority = uri
        path = '/'
    host, port = _splitport(authority)
    if default_port and port is None and (scheme is not None):
        dport = {'http': 80, 'https': 443}.get(scheme)
        if dport is not None:
            authority = '%s:%d' % (host, dport)
    return (authority, path)