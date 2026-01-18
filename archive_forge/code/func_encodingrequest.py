from __future__ import print_function
import collections
import contextlib
import gzip
import json
import logging
import sys
import time
import zlib
from datetime import datetime, timedelta
from io import BytesIO
from tornado import httputil
from tornado.web import RequestHandler
from urllib3.packages.six import binary_type, ensure_str
from urllib3.packages.six.moves.http_client import responses
from urllib3.packages.six.moves.urllib.parse import urlsplit
def encodingrequest(self, request):
    """Check for UA accepting gzip/deflate encoding"""
    data = b'hello, world!'
    encoding = request.headers.get('Accept-Encoding', '')
    headers = None
    if encoding == 'gzip':
        headers = [('Content-Encoding', 'gzip')]
        file_ = BytesIO()
        with contextlib.closing(gzip.GzipFile('', mode='w', fileobj=file_)) as zipfile:
            zipfile.write(data)
        data = file_.getvalue()
    elif encoding == 'deflate':
        headers = [('Content-Encoding', 'deflate')]
        data = zlib.compress(data)
    elif encoding == 'garbage-gzip':
        headers = [('Content-Encoding', 'gzip')]
        data = 'garbage'
    elif encoding == 'garbage-deflate':
        headers = [('Content-Encoding', 'deflate')]
        data = 'garbage'
    return Response(data, headers=headers)