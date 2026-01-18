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
def has_user_roles(self, roles: Union[Union[str, int, 'UserRole'], List[Union[str, int, 'UserRole']]], require_all: Optional[bool]=False) -> bool:
    """
        Checks against multiple roles
        """
    if not isinstance(roles, list):
        roles = [roles]
    valid_roles = [UserRole.parse_role(role) for role in roles]
    if require_all:
        return all((self.role >= role for role in valid_roles))
    return any((self.role >= role for role in valid_roles))