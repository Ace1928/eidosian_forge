import base64
import httpx
from lazyops.libs import lazyload
from typing import Optional, List, Dict, Any, Union, Iterable, Generator, AsyncGenerator
from .base import BaseModel
from .user_data import AZUserData
from .claims import APIKeyJWTClaims
@classmethod
def get_auth_token(cls, data: Union['Headers', Dict[str, str]], settings: Optional['AuthZeroSettings']=None) -> Optional[str]:
    """
        Returns the Auth Token from the Headers or Cookies
        """
    settings = settings or cls.get_settings()
    authorization_header_value = data.get(settings.authorization_header)
    if authorization_header_value:
        scheme, _, param = authorization_header_value.partition(' ')
        if scheme.lower() == settings.authorization_scheme and (not param.startswith('apikey:')):
            return param