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
def open_local_file(self, url):
    """Use local file."""
    import email.utils
    import mimetypes
    host, file = _splithost(url)
    localname = url2pathname(file)
    try:
        stats = os.stat(localname)
    except OSError as e:
        raise URLError(e.strerror, e.filename)
    size = stats.st_size
    modified = email.utils.formatdate(stats.st_mtime, usegmt=True)
    mtype = mimetypes.guess_type(url)[0]
    headers = email.message_from_string('Content-Type: %s\nContent-Length: %d\nLast-modified: %s\n' % (mtype or 'text/plain', size, modified))
    if not host:
        urlfile = file
        if file[:1] == '/':
            urlfile = 'file://' + file
        return addinfourl(open(localname, 'rb'), headers, urlfile)
    host, port = _splitport(host)
    if not port and socket.gethostbyname(host) in (localhost(),) + thishost():
        urlfile = file
        if file[:1] == '/':
            urlfile = 'file://' + file
        elif file[:2] == './':
            raise ValueError('local file url may start with / or file:. Unknown url of type: %s' % url)
        return addinfourl(open(localname, 'rb'), headers, urlfile)
    raise URLError('local file error: not on local host')