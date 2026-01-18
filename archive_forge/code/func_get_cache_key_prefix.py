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
def get_cache_key_prefix(self) -> str:
    """
        Returns the Cache Key Prefix
        """
    if self.cache_key_prefix:
        return self.cache_key_prefix
    if self.app_ingress:
        return get_hashed_key(self.app_domain)[:8]
    if self.app_name and self.app_env:
        return get_hashed_key(f'{self.app_name}-{self.app_env.name}')[:8]
    raise ValueError('Unable to determine cache key prefix, please set `app_name` and `app_env` or `app_ingress`')