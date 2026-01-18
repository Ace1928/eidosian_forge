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
def specific_method(self, request):
    """Confirm that the request matches the desired method type"""
    method = request.params.get('method')
    if method and (not isinstance(method, str)):
        method = method.decode('utf8')
    if request.method != method:
        return Response('Wrong method: %s != %s' % (method, request.method), status='400 Bad Request')
    return Response()