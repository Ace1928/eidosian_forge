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
def proxy_open(self, req, proxy, type):
    orig_type = req.type
    proxy_type, user, password, hostport = _parse_proxy(proxy)
    if proxy_type is None:
        proxy_type = orig_type
    if req.host and proxy_bypass(req.host):
        return None
    if user and password:
        user_pass = '%s:%s' % (unquote(user), unquote(password))
        creds = base64.b64encode(user_pass.encode()).decode('ascii')
        req.add_header('Proxy-authorization', 'Basic ' + creds)
    hostport = unquote(hostport)
    req.set_proxy(hostport, proxy_type)
    if orig_type == proxy_type or orig_type == 'https':
        return None
    else:
        return self.parent.open(req, timeout=req.timeout)