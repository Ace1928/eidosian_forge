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
def generate_secret_key() -> str:
    """ Generate a new securely-generated secret key appropriate for SHA-256
    HMAC signatures.

    This key could be used to sign Bokeh server session IDs, for example.
    """
    return _get_random_string()