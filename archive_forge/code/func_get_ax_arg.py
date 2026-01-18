import base64
import binascii
import hashlib
import hmac
import time
import urllib.parse
import uuid
import warnings
from tornado import httpclient
from tornado import escape
from tornado.httputil import url_concat
from tornado.util import unicode_type
from tornado.web import RequestHandler
from typing import List, Any, Dict, cast, Iterable, Union, Optional
def get_ax_arg(uri: str) -> str:
    if not ax_ns:
        return ''
    prefix = 'openid.' + ax_ns + '.type.'
    ax_name = None
    for name in handler.request.arguments.keys():
        if handler.get_argument(name) == uri and name.startswith(prefix):
            part = name[len(prefix):]
            ax_name = 'openid.' + ax_ns + '.value.' + part
            break
    if not ax_name:
        return ''
    return handler.get_argument(ax_name, '')