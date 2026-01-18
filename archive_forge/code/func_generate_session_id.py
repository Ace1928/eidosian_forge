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
def generate_session_id(secret_key: bytes | None=settings.secret_key_bytes(), signed: bool=settings.sign_sessions()) -> ID:
    """ Generate a random session ID.

    Typically, each browser tab connected to a Bokeh application has its own
    session ID. In production deployments of a Bokeh app, session IDs should be
    random and unguessable - otherwise users of the app could interfere with one
    another.
    """
    session_id = _get_random_string()
    if signed:
        session_id = '.'.join([session_id, _signature(session_id, secret_key)])
    return ID(session_id)