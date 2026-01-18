from __future__ import annotations
import uuid
import hashlib
import base64
from lazyops.libs.pooler import ThreadPooler
from lazyops.imports._pycryptodome import resolve_pycryptodome
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from typing import Any, Optional, List, Dict
def encode_params_to_url(params: Dict[str, Any], url: str) -> str:
    """
    Encodes the params to a url
    """
    url_parts = list(urlparse(url))
    query = dict(parse_qsl(url_parts[4]))
    query.update(params)
    url_parts[4] = urlencode(query)
    return urlunparse(url_parts)