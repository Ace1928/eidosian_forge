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
def import_assets(self, *path_parts, model: Optional[Type['BaseModel']]=None, load_file: Optional[bool]=False, **kwargs) -> Union[Dict[str, Any], List[Any]]:
    """
        Import assets from a module.

        args:
            path_parts: path parts to the assets directory (default: [])
            model: model to parse the assets with (default: None)
            load_file: load the file (default: False)
            **kwargs: additional arguments to pass to import_module_assets
        """
    import_assets = import_assets_func(self.module_name, 'assets')
    return import_assets(*path_parts, model=model, load_file=load_file, **kwargs)