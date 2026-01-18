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
def get_signed_cookie_key_version(self, name: str, value: Optional[str]=None) -> Optional[int]:
    """Returns the signing key version of the secure cookie.

        The version is returned as int.

        .. versionchanged:: 6.3

           Renamed from ``get_secure_cookie_key_version`` to
           ``set_signed_cookie_key_version`` to avoid confusion with other
           uses of "secure" in cookie attributes and prefixes. The old name
           remains as an alias.

        """
    self.require_setting('cookie_secret', 'secure cookies')
    if value is None:
        value = self.get_cookie(name)
    if value is None:
        return None
    return get_signature_key_version(value)