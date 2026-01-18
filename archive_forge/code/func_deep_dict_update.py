import re
import warnings
from dataclasses import is_dataclass
from typing import (
from weakref import WeakKeyDictionary
import fastapi
from fastapi._compat import (
from fastapi.datastructures import DefaultPlaceholder, DefaultType
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo
from typing_extensions import Literal
def deep_dict_update(main_dict: Dict[Any, Any], update_dict: Dict[Any, Any]) -> None:
    for key, value in update_dict.items():
        if key in main_dict and isinstance(main_dict[key], dict) and isinstance(value, dict):
            deep_dict_update(main_dict[key], value)
        elif key in main_dict and isinstance(main_dict[key], list) and isinstance(update_dict[key], list):
            main_dict[key] = main_dict[key] + update_dict[key]
        else:
            main_dict[key] = value