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
def get_whoami_data(self, include_data: Optional[bool]=None, **kwargs) -> Dict[str, Any]:
    """
        Returns the Whoami Data
        """
    include_data = include_data if include_data is not None else self.settings.is_development_env
    data = {'user': self.user_id, 'api_key': self.api_key, 'request_id': self.request_id}
    if include_data:
        data['data'] = self.model_dump(mode='json', **kwargs)
        if self.session:
            data['session_ttl'] = self.session.ttl
        data['domain_source'] = self.domain_source
    if self.settings.is_development_env and self.session:
        data['session'] = self.session.model_dump(mode='json')
    return data