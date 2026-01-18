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
def decode_response_body(response):
    """
    Decodes the JSON-format response body

    Arguments
    ---------
    response: tornado.httpclient.HTTPResponse

    Returns
    -------
    Decoded response content
    """
    try:
        body = codecs.decode(response.body, 'ascii')
    except Exception:
        body = codecs.decode(response.body, 'utf-8')
    body = re.sub('"', '"', body)
    body = re.sub("'", '"', body)
    body = json.loads(body)
    return body