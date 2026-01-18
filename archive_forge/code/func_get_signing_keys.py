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
def get_signing_keys(self, refresh: bool=False) -> List[PyJWK]:
    jwk_set = self.get_jwk_set(refresh)
    signing_keys = [jwk_set_key for jwk_set_key in jwk_set.keys if jwk_set_key.public_key_use in ['sig', None] and jwk_set_key.key_id]
    if not signing_keys:
        raise PyJWKClientError('The JWKS endpoint did not contain any signing keys')
    return signing_keys