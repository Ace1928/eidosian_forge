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
def render_linked_js(self, js_files: Iterable[str]) -> str:
    """Default method used to render the final js links for the
        rendered webpage.

        Override this method in a sub-classed controller to change the output.
        """
    paths = []
    unique_paths = set()
    for path in js_files:
        if not is_absolute(path):
            path = self.static_url(path)
        if path not in unique_paths:
            paths.append(path)
            unique_paths.add(path)
    return ''.join(('<script src="' + escape.xhtml_escape(p) + '" type="text/javascript"></script>' for p in paths))