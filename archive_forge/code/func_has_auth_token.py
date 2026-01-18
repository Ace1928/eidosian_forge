import base64
import httpx
from lazyops.libs import lazyload
from typing import Optional, List, Dict, Any, Union, Iterable, Generator, AsyncGenerator
from .base import BaseModel
from .user_data import AZUserData
from .claims import APIKeyJWTClaims
@property
def has_auth_token(self) -> bool:
    """
        Returns True if the Auth Token is Present
        """
    return bool(self.auth_token)