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
def generate_unique_id(route: 'APIRoute') -> str:
    operation_id = f'{route.name}{route.path_format}'
    operation_id = re.sub('\\W', '_', operation_id)
    assert route.methods
    operation_id = f'{operation_id}_{list(route.methods)[0].lower()}'
    return operation_id