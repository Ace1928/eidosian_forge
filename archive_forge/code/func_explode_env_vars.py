import os
import warnings
from pathlib import Path
from typing import AbstractSet, Any, Callable, ClassVar, Dict, List, Mapping, Optional, Tuple, Type, Union
from .config import BaseConfig, Extra
from .fields import ModelField
from .main import BaseModel
from .types import JsonWrapper
from .typing import StrPath, display_as_type, get_origin, is_union
from .utils import deep_update, lenient_issubclass, path_type, sequence_like
def explode_env_vars(self, field: ModelField, env_vars: Mapping[str, Optional[str]]) -> Dict[str, Any]:
    """
        Process env_vars and extract the values of keys containing env_nested_delimiter into nested dictionaries.

        This is applied to a single field, hence filtering by env_var prefix.
        """
    prefixes = [f'{env_name}{self.env_nested_delimiter}' for env_name in field.field_info.extra['env_names']]
    result: Dict[str, Any] = {}
    for env_name, env_val in env_vars.items():
        if not any((env_name.startswith(prefix) for prefix in prefixes)):
            continue
        env_name_without_prefix = env_name[self.env_prefix_len:]
        _, *keys, last_key = env_name_without_prefix.split(self.env_nested_delimiter)
        env_var = result
        for key in keys:
            env_var = env_var.setdefault(key, {})
        env_var[last_key] = env_val
    return result