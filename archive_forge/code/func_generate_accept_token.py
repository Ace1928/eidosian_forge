import base64
import hashlib
import os
from typing import Dict, List, Optional, Union
from h11._headers import Headers as H11Headers
from .events import Event
from .typing import Headers
def generate_accept_token(token: bytes) -> bytes:
    accept_token = token + ACCEPT_GUID
    accept_token = hashlib.sha1(accept_token).digest()
    return base64.b64encode(accept_token)