import base64
import httpx
from lazyops.libs import lazyload
from typing import Optional, List, Dict, Any, Union, Iterable, Generator, AsyncGenerator
from .base import BaseModel
from .user_data import AZUserData
from .claims import APIKeyJWTClaims
@classmethod
def get_x_api_key(cls, data: Union['Headers', Dict[str, str]], settings: Optional['AuthZeroSettings']=None) -> Optional[str]:
    """
        Returns the API Key from the Headers or Cookies
        """
    settings = settings or cls.get_settings()
    if (api_key := data.get(settings.apikey_header)):
        return api_key
    authorization_header_value = data.get(settings.authorization_header)
    if authorization_header_value:
        scheme, _, param = authorization_header_value.partition(' ')
        if scheme.lower() in {'basic', 'bearer'}:
            if scheme.lower() == 'basic':
                param = base64.b64decode(param).decode('utf-8')
            if 'apikey:' not in param:
                return None
            _, api_key = param.split(':', 1)
            return api_key