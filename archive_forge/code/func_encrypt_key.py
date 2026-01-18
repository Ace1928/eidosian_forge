from __future__ import annotations
import uuid
import hashlib
import base64
from lazyops.libs.pooler import ThreadPooler
from lazyops.imports._pycryptodome import resolve_pycryptodome
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from typing import Any, Optional, List, Dict
def encrypt_key(key: str, secret_key: str, access_key: str) -> str:
    """
    Encrypts the Key
    """
    resolve_pycryptodome(True)
    from Cryptodome.Cipher import AES
    cipher = AES.new(secret_key.encode(), AES.MODE_CBC, access_key.encode())
    string = cipher.encrypt(resize_key(key).encode())
    string = ''.join(('{:02x}'.format(c) for c in string))
    return string