import typing
from lazyops.utils.imports import resolve_missing
import inspect
import pkg_resources
from pathlib import Path
from pydantic import BaseModel
from pydantic.fields import FieldInfo
@property
def module_config_path(self) -> Path:
    """
        Returns the config module path
        """
    return Path(inspect.getfile(self.__class__)).parent