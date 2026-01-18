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
def create_get_app_default_file(configs_path: Path, module_name: Optional[str]=None) -> Callable[[Optional[str], Optional[bool], Optional[str]], Optional[Path]]:
    """
    Creates a get_app_default_file wrapper
    """
    kwargs = {}
    if module_name is not None:
        kwargs['module_name'] = module_name
    return functools.partial(get_app_default_file, configs_path, **kwargs)