import base64
import httpx
from lazyops.libs import lazyload
from typing import Optional, List, Dict, Any, Union, Iterable, Generator, AsyncGenerator
from .base import BaseModel
from .user_data import AZUserData
from .claims import APIKeyJWTClaims
@property
def has_both_auth(self) -> bool:
    """
        Returns True if the Auth Token and X-API-Key are Present
        """
    return self.has_auth_token and self.has_x_api_key