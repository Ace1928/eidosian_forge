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
def http_error_auth_reqed(self, auth_header, host, req, headers):
    authreq = headers.get(auth_header, None)
    if self.retried > 5:
        raise HTTPError(req.full_url, 401, 'digest auth failed', headers, None)
    else:
        self.retried += 1
    if authreq:
        scheme = authreq.split()[0]
        if scheme.lower() == 'digest':
            return self.retry_http_digest_auth(req, authreq)
        elif scheme.lower() != 'basic':
            raise ValueError("AbstractDigestAuthHandler does not support the following scheme: '%s'" % scheme)