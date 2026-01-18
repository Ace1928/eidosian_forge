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
def get_token_payload(token: str) -> TokenPayload:
    """Extract the payload from the token.

    Args:
        token (str):
            A JWT token containing the session_id and other data.

    Returns:
        dict
    """
    decoded = json.loads(_base64_decode(token.split('.')[0]))
    if _TOKEN_ZLIB_KEY in decoded:
        decompressed = zlib.decompress(_base64_decode(decoded[_TOKEN_ZLIB_KEY]))
        del decoded[_TOKEN_ZLIB_KEY]
        decoded.update(json.loads(decompressed, cls=_BytesDecoder))
    del decoded['session_id']
    return decoded