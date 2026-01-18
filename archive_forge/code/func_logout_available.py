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
@property
def logout_available(self):
    """Whether a LogoutHandler is needed."""
    return True