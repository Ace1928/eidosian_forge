import base64
import httpx
from lazyops.libs import lazyload
from typing import Optional, List, Dict, Any, Union, Iterable, Generator, AsyncGenerator
from .base import BaseModel
from .user_data import AZUserData
from .claims import APIKeyJWTClaims
@property
def has_x_api_key(self) -> bool:
    """
        Returns True if the X-API-Key is Present
        """
    return bool(self.x_api_key)