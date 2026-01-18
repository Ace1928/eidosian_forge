from __future__ import annotations
import functools
import inspect
import ipaddress
import json
import mimetypes
import os
import re
import types
import warnings
from http.client import responses
from logging import Logger
from typing import TYPE_CHECKING, Any, Awaitable, Coroutine, Sequence, cast
from urllib.parse import urlparse
import prometheus_client
from jinja2 import TemplateNotFound
from jupyter_core.paths import is_hidden
from jupyter_events import EventLogger
from tornado import web
from tornado.log import app_log
from traitlets.config import Application
import jupyter_server
from jupyter_server import CallContext
from jupyter_server._sysinfo import get_sys_info
from jupyter_server._tz import utcnow
from jupyter_server.auth.decorator import allow_unauthenticated, authorized
from jupyter_server.auth.identity import User
from jupyter_server.i18n import combine_translations
from jupyter_server.services.security import csp_report_uri
from jupyter_server.utils import (
def check_referer(self) -> bool:
    """Check Referer for cross-site requests.
        Disables requests to certain endpoints with
        external or missing Referer.
        If set, allow_origin settings are applied to the Referer
        to whitelist specific cross-origin sites.
        Used on GET for api endpoints and /files/
        to block cross-site inclusion (XSSI).
        """
    if self.allow_origin == '*' or self.skip_check_origin():
        return True
    host = self.request.headers.get('Host')
    referer = self.request.headers.get('Referer')
    if not host:
        self.log.warning('Blocking request with no host')
        return False
    if not referer:
        self.log.warning('Blocking request with no referer')
        return False
    referer_url = urlparse(referer)
    referer_host = referer_url.netloc
    if referer_host == host:
        return True
    origin = f'{referer_url.scheme}://{referer_url.netloc}'
    if self.allow_origin:
        allow = self.allow_origin == origin
    elif self.allow_origin_pat:
        allow = bool(re.match(self.allow_origin_pat, origin))
    else:
        allow = False
    if not allow:
        self.log.warning('Blocking Cross Origin request for %s.  Referer: %s, Host: %s', self.request.path, origin, host)
    return allow