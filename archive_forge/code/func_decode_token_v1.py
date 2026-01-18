from __future__ import annotations
from pydantic import ValidationError
from lazyops.libs import lazyload
from lazyops.libs.pooler import ThreadPooler
from .lazy import get_az_settings, get_az_resource
from typing import Any, Optional, List
def decode_token_v1(request: Optional['Request']=None, token: Optional[str]=None, settings: Optional['AuthZeroSettings']=None) -> 'UserJWTClaims':
    """
    Attempts to decode the token from the request
    """
    from jose import jwt, JWTError
    try:
        assert request or token, 'Either a request or token must be provided'
        settings = settings or get_az_settings()
        token = token or get_auth_token(request.headers)
        claims = UserJWTClaims(**jwt.decode(token=token, key=settings.jwks, algorithms=['RS256'], audience=settings.audience, issuer=settings.issuer))
    except (JWTError, ValidationError) as e:
        from ..types.errors import InvalidTokenException
        raise InvalidTokenException(error=e) from e
    return claims