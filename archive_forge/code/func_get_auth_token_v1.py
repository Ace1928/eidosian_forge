from __future__ import annotations
from pydantic import ValidationError
from lazyops.libs import lazyload
from lazyops.libs.pooler import ThreadPooler
from .lazy import get_az_settings, get_az_resource
from typing import Any, Optional, List
def get_auth_token_v1(headers: 'Headers'):
    """
    Gets the Auth Token from the Headers
    """
    authorization_header_value = headers.get('Authorization')
    if authorization_header_value:
        scheme, _, param = authorization_header_value.partition(' ')
        if scheme.lower() == 'bearer':
            return param
    from ..types.errors import NoTokenException
    raise NoTokenException(detail='Invalid Authorization Header')