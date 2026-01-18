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
def create_signed_value(self, name: str, value: Union[str, bytes], version: Optional[int]=None) -> bytes:
    """Signs and timestamps a string so it cannot be forged.

        Normally used via set_signed_cookie, but provided as a separate
        method for non-cookie uses.  To decode a value not stored
        as a cookie use the optional value argument to get_signed_cookie.

        .. versionchanged:: 3.2.1

           Added the ``version`` argument.  Introduced cookie version 2
           and made it the default.
        """
    self.require_setting('cookie_secret', 'secure cookies')
    secret = self.application.settings['cookie_secret']
    key_version = None
    if isinstance(secret, dict):
        if self.application.settings.get('key_version') is None:
            raise Exception('key_version setting must be used for secret_key dicts')
        key_version = self.application.settings['key_version']
    return create_signed_value(secret, name, value, version=version, key_version=key_version)