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
def multi_redirect(self, request):
    """Performs a redirect chain based on ``redirect_codes``"""
    codes = request.params.get('redirect_codes', b'200').decode('utf-8')
    head, tail = codes.split(',', 1) if ',' in codes else (codes, None)
    status = '{0} {1}'.format(head, responses[int(head)])
    if not tail:
        return Response('Done redirecting', status=status)
    headers = [('Location', '/multi_redirect?redirect_codes=%s' % tail)]
    return Response(status=status, headers=headers)