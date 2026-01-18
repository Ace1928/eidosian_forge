import typing
from lazyops.utils.imports import resolve_missing
import inspect
import pkg_resources
from pathlib import Path
from pydantic import BaseModel
from pydantic.fields import FieldInfo
@property
def module_pkg_name(self) -> str:
    """
        Returns the module pkg name
        
        {pkg}/src   -> src
        {pkg}/{pkg} -> {pkg}
        """
    config_path = self.module_config_path.as_posix()
    module_path = self.module_path.as_posix()
    return config_path.replace(module_path, '').strip().split('/', 2)[1]