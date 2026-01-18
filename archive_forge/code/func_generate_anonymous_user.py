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
def generate_anonymous_user(self, handler: web.RequestHandler) -> User:
    """Generate a random anonymous user.

        For use when a single shared token is used,
        but does not identify a user.
        """
    user_id = uuid.uuid4().hex
    moon = get_anonymous_username()
    name = display_name = f'Anonymous {moon}'
    initials = f'A{moon[0]}'
    color = None
    handler.log.debug(f'Generating new user for token-authenticated request: {user_id}')
    return User(user_id, name, display_name, initials, None, color)