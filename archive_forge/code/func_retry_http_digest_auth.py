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
def retry_http_digest_auth(self, req, auth):
    token, challenge = auth.split(' ', 1)
    chal = parse_keqv_list(filter(None, parse_http_list(challenge)))
    auth = self.get_authorization(req, chal)
    if auth:
        auth_val = 'Digest %s' % auth
        if req.headers.get(self.auth_header, None) == auth_val:
            return None
        req.add_unredirected_header(self.auth_header, auth_val)
        resp = self.parent.open(req, timeout=req.timeout)
        return resp