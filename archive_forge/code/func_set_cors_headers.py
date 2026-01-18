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
def set_cors_headers(self) -> None:
    """Add CORS headers, if defined

        Now that current_user is async (jupyter-server 2.0),
        must be called at the end of prepare(), instead of in set_default_headers.
        """
    if self.allow_origin:
        self.set_header('Access-Control-Allow-Origin', self.allow_origin)
    elif self.allow_origin_pat:
        origin = self.get_origin()
        if origin and re.match(self.allow_origin_pat, origin):
            self.set_header('Access-Control-Allow-Origin', origin)
    elif self.token_authenticated and 'Access-Control-Allow-Origin' not in self.settings.get('headers', {}):
        self.set_header('Access-Control-Allow-Origin', self.request.headers.get('Origin', ''))
    if self.allow_credentials:
        self.set_header('Access-Control-Allow-Credentials', 'true')