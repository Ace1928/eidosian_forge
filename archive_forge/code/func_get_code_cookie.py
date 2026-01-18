import asyncio
import base64
import codecs
import datetime as dt
import hashlib
import json
import logging
import os
import re
import urllib.parse as urlparse
import uuid
from base64 import urlsafe_b64encode
from functools import partial
import tornado
from bokeh.server.auth_provider import AuthProvider
from tornado.auth import OAuth2Mixin
from tornado.httpclient import HTTPError as HTTPClientError, HTTPRequest
from tornado.web import HTTPError, RequestHandler, decode_signed_value
from tornado.websocket import WebSocketHandler
from .config import config
from .entry_points import entry_points_for
from .io.resources import (
from .io.state import state
from .util import base64url_encode, decode_token
def get_code_cookie(self):
    code = (self.get_secure_cookie(CODE_COOKIE_NAME, max_age_days=config.oauth_expiry) or b'').decode('utf8', 'replace')
    self.clear_cookie(CODE_COOKIE_NAME)
    return code