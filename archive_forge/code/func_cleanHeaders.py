from your HTTP server.
import pprint
import re
import socket
import sys
import time
import traceback
import os
import json
import unittest  # pylint: disable=deprecated-module,preferred-module
import warnings
import functools
import http.client
import urllib.parse
from more_itertools.more import always_iterable
import jaraco.functools
def cleanHeaders(headers, method, body, host, port):
    """Return request headers, with required headers added (if missing)."""
    if headers is None:
        headers = []
    found = False
    for k, _v in headers:
        if k.lower() == 'host':
            found = True
            break
    if not found:
        if port == 80:
            headers.append(('Host', host))
        else:
            headers.append(('Host', '%s:%s' % (host, port)))
    if method in methods_with_bodies:
        found = False
        for k, v in headers:
            if k.lower() == 'content-type':
                found = True
                break
        if not found:
            headers.append(('Content-Type', 'application/x-www-form-urlencoded'))
            headers.append(('Content-Length', str(len(body or ''))))
    return headers