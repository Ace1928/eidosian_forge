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
def get_browser_locale(self, default: str='en_US') -> tornado.locale.Locale:
    """Determines the user's locale from ``Accept-Language`` header.

        See http://www.w3.org/Protocols/rfc2616/rfc2616-sec14.html#sec14.4
        """
    if 'Accept-Language' in self.request.headers:
        languages = self.request.headers['Accept-Language'].split(',')
        locales = []
        for language in languages:
            parts = language.strip().split(';')
            if len(parts) > 1 and parts[1].strip().startswith('q='):
                try:
                    score = float(parts[1].strip()[2:])
                    if score < 0:
                        raise ValueError()
                except (ValueError, TypeError):
                    score = 0.0
            else:
                score = 1.0
            if score > 0:
                locales.append((parts[0], score))
        if locales:
            locales.sort(key=lambda pair: pair[1], reverse=True)
            codes = [loc[0] for loc in locales]
            return locale.get(*codes)
    return locale.get(default)