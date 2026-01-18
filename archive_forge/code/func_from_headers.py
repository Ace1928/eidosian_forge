import base64
import httpx
from lazyops.libs import lazyload
from typing import Optional, List, Dict, Any, Union, Iterable, Generator, AsyncGenerator
from .base import BaseModel
from .user_data import AZUserData
from .claims import APIKeyJWTClaims
@classmethod
def from_headers(cls, headers: 'Headers', settings: Optional['AuthZeroSettings']=None) -> 'AuthObject':
    """
        Returns an AuthObject from the Headers
        """
    settings = settings or cls.get_settings()
    return cls(auth_token=cls.get_auth_token(data=headers, settings=settings), x_api_key=cls.get_x_api_key(data=headers, settings=settings))