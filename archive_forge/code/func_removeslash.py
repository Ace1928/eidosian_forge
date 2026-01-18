import base64
import binascii
import datetime
import email.utils
import functools
import gzip
import hashlib
import hmac
import http.cookies
from inspect import isclass
from io import BytesIO
import mimetypes
import numbers
import os.path
import re
import socket
import sys
import threading
import time
import warnings
import tornado
import traceback
import types
import urllib.parse
from urllib.parse import urlencode
from tornado.concurrent import Future, future_set_result_unless_cancelled
from tornado import escape
from tornado import gen
from tornado.httpserver import HTTPServer
from tornado import httputil
from tornado import iostream
from tornado import locale
from tornado.log import access_log, app_log, gen_log
from tornado import template
from tornado.escape import utf8, _unicode
from tornado.routing import (
from tornado.util import ObjectDict, unicode_type, _websocket_mask
from typing import (
from types import TracebackType
import typing
def removeslash(method: Callable[..., Optional[Awaitable[None]]]) -> Callable[..., Optional[Awaitable[None]]]:
    """Use this decorator to remove trailing slashes from the request path.

    For example, a request to ``/foo/`` would redirect to ``/foo`` with this
    decorator. Your request handler mapping should use a regular expression
    like ``r'/foo/*'`` in conjunction with using the decorator.
    """

    @functools.wraps(method)
    def wrapper(self: RequestHandler, *args, **kwargs) -> Optional[Awaitable[None]]:
        if self.request.path.endswith('/'):
            if self.request.method in ('GET', 'HEAD'):
                uri = self.request.path.rstrip('/')
                if uri:
                    if self.request.query:
                        uri += '?' + self.request.query
                    self.redirect(uri, permanent=True)
                    return None
            else:
                raise HTTPError(404)
        return method(self, *args, **kwargs)
    return wrapper