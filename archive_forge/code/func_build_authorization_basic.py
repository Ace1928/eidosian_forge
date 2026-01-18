from __future__ import annotations
import base64
import binascii
import ipaddress
import re
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, cast
from . import exceptions
from .typing import (
def build_authorization_basic(username: str, password: str) -> str:
    """
    Build an ``Authorization`` header for HTTP Basic Auth.

    This is the reverse of :func:`parse_authorization_basic`.

    """
    assert ':' not in username
    user_pass = f'{username}:{password}'
    basic_credentials = base64.b64encode(user_pass.encode()).decode()
    return 'Basic ' + basic_credentials