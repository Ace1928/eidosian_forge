from __future__ import annotations
import binascii
import datetime
import json
import os
import re
import sys
import typing as t
import uuid
from dataclasses import asdict, dataclass
from http.cookies import Morsel
from tornado import escape, httputil, web
from traitlets import Bool, Dict, Type, Unicode, default
from traitlets.config import LoggingConfigurable
from jupyter_server.transutils import _i18n
from .security import passwd_check, set_password
from .utils import get_anonymous_username
def clear_login_cookie(self, handler: web.RequestHandler) -> None:
    """Clear the login cookie, effectively logging out the session."""
    cookie_options = {}
    cookie_options.update(self.cookie_options)
    path = cookie_options.setdefault('path', handler.base_url)
    cookie_name = self.get_cookie_name(handler)
    handler.clear_cookie(cookie_name, path=path)
    if path and path != '/':
        self._force_clear_cookie(handler, cookie_name)