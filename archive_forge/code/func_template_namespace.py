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
@property
def template_namespace(self) -> dict[str, Any]:
    return dict(base_url=self.base_url, default_url=self.default_url, ws_url=self.ws_url, logged_in=self.logged_in, allow_password_change=getattr(self.identity_provider, 'allow_password_change', False), auth_enabled=self.identity_provider.auth_enabled, login_available=self.identity_provider.login_available, token_available=bool(self.token), static_url=self.static_url, sys_info=json_sys_info(), contents_js_source=self.contents_js_source, version_hash=self.version_hash, xsrf_form_html=self.xsrf_form_html, token=self.token, xsrf_token=self.xsrf_token.decode('utf8'), nbjs_translations=json.dumps(combine_translations(self.request.headers.get('Accept-Language', ''))), **self.jinja_template_vars)