from __future__ import annotations
import base64
import binascii
import ipaddress
import re
from typing import Callable, List, Optional, Sequence, Tuple, TypeVar, cast
from . import exceptions
from .typing import (
def build_www_authenticate_basic(realm: str) -> str:
    """
    Build a ``WWW-Authenticate`` header for HTTP Basic Auth.

    Args:
        realm: identifier of the protection space.

    """
    realm = build_quoted_string(realm)
    charset = build_quoted_string('UTF-8')
    return f'Basic realm={realm}, charset={charset}'