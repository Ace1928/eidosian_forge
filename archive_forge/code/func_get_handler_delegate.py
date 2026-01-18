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
def get_handler_delegate(self, request: httputil.HTTPServerRequest, target_class: Type[RequestHandler], target_kwargs: Optional[Dict[str, Any]]=None, path_args: Optional[List[bytes]]=None, path_kwargs: Optional[Dict[str, bytes]]=None) -> '_HandlerDelegate':
    """Returns `~.httputil.HTTPMessageDelegate` that can serve a request
        for application and `RequestHandler` subclass.

        :arg httputil.HTTPServerRequest request: current HTTP request.
        :arg RequestHandler target_class: a `RequestHandler` class.
        :arg dict target_kwargs: keyword arguments for ``target_class`` constructor.
        :arg list path_args: positional arguments for ``target_class`` HTTP method that
            will be executed while handling a request (``get``, ``post`` or any other).
        :arg dict path_kwargs: keyword arguments for ``target_class`` HTTP method.
        """
    return _HandlerDelegate(self, request, target_class, target_kwargs, path_args, path_kwargs)