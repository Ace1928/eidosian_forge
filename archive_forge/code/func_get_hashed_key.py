from __future__ import annotations
import uuid
import hashlib
import base64
from lazyops.libs.pooler import ThreadPooler
from lazyops.imports._pycryptodome import resolve_pycryptodome
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from typing import Any, Optional, List, Dict
def get_hashed_key(key: Any) -> str:
    """
    Returns a Hashed Key
    """
    return hashlib.sha256(str(key).encode()).hexdigest()