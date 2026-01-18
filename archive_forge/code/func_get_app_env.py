import os
import functools
from enum import Enum
from pathlib import Path
from lazyops.types.models import BaseSettings, pre_root_validator, validator
from lazyops.imports._pydantic import BaseAppSettings, BaseModel
from lazyops.utils.system import is_in_kubernetes, get_host_name
from lazyops.utils.assets import create_get_assets_wrapper, create_import_assets_wrapper
from lazyops.libs.fastapi_utils import GlobalContext
from lazyops.libs.fastapi_utils.types.persistence import TemporaryData
from typing import List, Optional, Dict, Any, Callable, Union, Type, TYPE_CHECKING
def get_app_env(name: Optional[str]=None) -> AppEnv:
    """
    Retrieves the app environment
    """
    for key in {'SERVER_ENV', f'{name.upper()}_ENV' if name is not None else 'APP_SERVER_ENV', 'APP_ENV', 'ENVIRONMENT'}:
        if (env_value := os.getenv(key)):
            return AppEnv.from_env(env_value)
    if is_in_kubernetes():
        parts = get_host_name().split('-')
        return AppEnv.from_env(parts[2]) if len(parts) > 3 else AppEnv.PRODUCTION
    return AppEnv.LOCAL