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
def identity_provider(self) -> IdentityProvider:
    if 'identity_provider' not in self.settings:
        warnings.warn("The Tornado web application does not have an 'identity_provider' defined in its settings. In future releases of jupyter_server, this will be a required key for all subclasses of `JupyterHandler`. For an example, see the jupyter_server source code for how to add an identity provider to the tornado settings: https://github.com/jupyter-server/jupyter_server/blob/v2.0.0/jupyter_server/serverapp.py#L242", stacklevel=2)
        from jupyter_server.auth import IdentityProvider
        self.settings['identity_provider'] = IdentityProvider(config=self.settings.get('config', None))
    return cast('IdentityProvider', self.settings['identity_provider'])