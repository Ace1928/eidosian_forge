from __future__ import annotations
import logging # isort:skip
import base64
import calendar
import codecs
import datetime as dt
import hashlib
import hmac
import json
import time
import zlib
from typing import TYPE_CHECKING, Any
from ..core.types import ID
from ..settings import settings
from .warnings import warn
def get_session_id(token: str) -> ID:
    """Extracts the session id from a JWT token.

    Args:
        token (str):
            A JWT token containing the session_id and other data.

    Returns:
       str
    """
    decoded = json.loads(_base64_decode(token.split('.')[0]))
    return decoded['session_id']