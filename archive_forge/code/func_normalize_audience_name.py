from __future__ import annotations
import uuid
import hashlib
import base64
from lazyops.libs.pooler import ThreadPooler
from lazyops.imports._pycryptodome import resolve_pycryptodome
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from typing import Any, Optional, List, Dict
def normalize_audience_name(name: str) -> str:
    """
    Normalizes the audience name to transform the url

    >>> normalize_audience_name('https://domain.us.auth0.com/userinfo')
    'domain-us-auth0-com-userinfo'
    """
    return name.replace('https://', '').replace('http://', '').replace('/', '-').replace('.', '-').lower()