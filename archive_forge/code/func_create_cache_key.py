from __future__ import annotations
import os
import time
from pathlib import Path
from functools import lru_cache
from lazyops.utils.logs import logger as _logger, null_logger as _null_logger, Logger
from lazyops.imports._pydantic import BaseSettings
from lazyops.libs import lazyload
from lazyops.libs.proxyobj import ProxyObject
from lazyops.libs.abcs.configs.types import AppEnv
from lazyops.libs.fastapi_utils.types.persistence import TemporaryData
from pydantic import model_validator, computed_field, Field
from ..types.user_roles import UserRole
from ..utils.helpers import get_hashed_key, encrypt_key, decrypt_key, aencrypt_key, adecrypt_key, normalize_audience_name
from typing import List, Optional, Dict, Any, Union, overload, Callable, Tuple, TYPE_CHECKING
def create_cache_key(self, prefix: Optional[str]=None, suffix: Optional[str]=None, include_client_id: Optional[bool]=True, kind: Optional[str]=None) -> str:
    """
        Creates a Cache Key based on the Prefix, Suffix, and Kind
        """
    cache_key_prefix = None
    if prefix is not None:
        cache_key_prefix = prefix
    elif kind is not None:
        cache_key_prefix = f'{self.get_cache_key_prefix()}.{kind}'
    else:
        cache_key_prefix = self.get_cache_key_prefix()
    if include_client_id:
        cache_key_prefix = f'{cache_key_prefix}.{self.client_id[-10:]}'
    if suffix is not None:
        cache_key_prefix = f'{cache_key_prefix}.{suffix}'
    return cache_key_prefix