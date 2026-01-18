from __future__ import annotations
from pydantic import Field, model_validator, PrivateAttr
from lazyops.types.models import ConfigDict, schema_extra
from lazyops.libs import lazyload
from .base import BaseModel
from .common import UserType, ValidationMethod
from .user_roles import UserRole
from .user_session import UserSession
from .user_data import AZUserData
from .claims import UserJWTClaims, APIKeyJWTClaims
from .auth import AuthObject
from .security import Authorization, APIKey
from .errors import (
from ..utils.lazy import logger, ThreadPooler
from ..utils.helpers import parse_scopes, get_hashed_key
from ..utils.decoders import decode_token
from typing import Optional, List, Dict, Any, Union, Callable, Iterable, TYPE_CHECKING
def get_session_cookie_kwargs(self, is_delete: Optional[bool]=None) -> Optional[Dict[str, Any]]:
    """
        Gets the Session Cookie Value
        """
    cookie_kws = {'key': self.settings.session_cookie_key, 'httponly': True, 'secure': self.settings.is_secure_ingress}
    if not is_delete:
        cookie_kws['value'] = self.session_flow.cache_key
        cookie_kws['expires'] = self.session.expiration_ts
    return cookie_kws