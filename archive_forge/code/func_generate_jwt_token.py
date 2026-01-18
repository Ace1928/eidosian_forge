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
def generate_jwt_token(session_id: ID, secret_key: bytes | None=settings.secret_key_bytes(), signed: bool=settings.sign_sessions(), extra_payload: TokenPayload | None=None, expiration: int=300) -> str:
    """ Generates a JWT token given a session_id and additional payload.

    Args:
        session_id (str):
            The session id to add to the token

        secret_key (str, optional) :
            Secret key (default: value of BOKEH_SECRET_KEY environment variable)

        signed (bool, optional) :
            Whether to sign the session ID (default: value of BOKEH_SIGN_SESSIONS
            environment variable)

        extra_payload (dict, optional) :
            Extra key/value pairs to include in the Bokeh session token

        expiration (int, optional) :
            Expiration time

    Returns:
        str
    """
    now = calendar.timegm(dt.datetime.now(tz=dt.timezone.utc).timetuple())
    payload = {'session_id': session_id, 'session_expiry': now + expiration}
    if extra_payload:
        if 'session_id' in extra_payload:
            raise RuntimeError("extra_payload for session tokens may not contain 'session_id'")
        extra_payload_str = json.dumps(extra_payload, cls=_BytesEncoder).encode('utf-8')
        compressed = zlib.compress(extra_payload_str, level=9)
        payload[_TOKEN_ZLIB_KEY] = _base64_encode(compressed)
    token = _base64_encode(json.dumps(payload))
    secret_key = _ensure_bytes(secret_key)
    if not signed:
        return token
    return token + '.' + _signature(token, secret_key)