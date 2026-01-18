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
def check_session_id_signature(session_id: str, secret_key: bytes | None=settings.secret_key_bytes(), signed: bool | None=settings.sign_sessions()) -> bool:
    """Check the signature of a session ID, returning True if it's valid.

    The server uses this function to check whether a session ID was generated
    with the correct secret key. If signed sessions are disabled, this function
    always returns True.
    """
    secret_key = _ensure_bytes(secret_key)
    if signed:
        id_pieces = session_id.split('.', 1)
        if len(id_pieces) != 2:
            return False
        provided_id_signature = id_pieces[1]
        expected_id_signature = _signature(id_pieces[0], secret_key)
        return hmac.compare_digest(expected_id_signature, provided_id_signature)
    return True