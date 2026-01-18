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
def data_open(self, req):
    url = req.full_url
    scheme, data = url.split(':', 1)
    mediatype, data = data.split(',', 1)
    data = unquote_to_bytes(data)
    if mediatype.endswith(';base64'):
        data = base64.decodebytes(data)
        mediatype = mediatype[:-7]
    if not mediatype:
        mediatype = 'text/plain;charset=US-ASCII'
    headers = email.message_from_string('Content-type: %s\nContent-length: %d\n' % (mediatype, len(data)))
    return addinfourl(io.BytesIO(data), headers, url)