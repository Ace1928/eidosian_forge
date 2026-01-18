import json
import urllib.request
from functools import lru_cache
from ssl import SSLContext
from typing import Any, Dict, List, Optional
from urllib.error import URLError
from .api_jwk import PyJWK, PyJWKSet
from .api_jwt import decode_complete as decode_token
from .exceptions import PyJWKClientConnectionError, PyJWKClientError
from .jwk_set_cache import JWKSetCache
@staticmethod
def match_kid(signing_keys: List[PyJWK], kid: str) -> Optional[PyJWK]:
    signing_key = None
    for key in signing_keys:
        if key.key_id == kid:
            signing_key = key
            break
    return signing_key