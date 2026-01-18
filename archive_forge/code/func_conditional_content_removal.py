import json
import mimetypes
import os
import sys
from copy import copy
from functools import partial
from http import HTTPStatus
from importlib import import_module
from io import BytesIO, IOBase
from urllib.parse import unquote_to_bytes, urljoin, urlparse, urlsplit
from asgiref.sync import sync_to_async
from django.conf import settings
from django.core.handlers.asgi import ASGIRequest
from django.core.handlers.base import BaseHandler
from django.core.handlers.wsgi import LimitedStream, WSGIRequest
from django.core.serializers.json import DjangoJSONEncoder
from django.core.signals import got_request_exception, request_finished, request_started
from django.db import close_old_connections
from django.http import HttpHeaders, HttpRequest, QueryDict, SimpleCookie
from django.test import signals
from django.test.utils import ContextList
from django.urls import resolve
from django.utils.encoding import force_bytes
from django.utils.functional import SimpleLazyObject
from django.utils.http import urlencode
from django.utils.itercompat import is_iterable
from django.utils.regex_helper import _lazy_re_compile
def conditional_content_removal(request, response):
    """
    Simulate the behavior of most web servers by removing the content of
    responses for HEAD requests, 1xx, 204, and 304 responses. Ensure
    compliance with RFC 9112 Section 6.3.
    """
    if 100 <= response.status_code < 200 or response.status_code in (204, 304):
        if response.streaming:
            response.streaming_content = []
        else:
            response.content = b''
    if request.method == 'HEAD':
        if response.streaming:
            response.streaming_content = []
        else:
            response.content = b''
    return response