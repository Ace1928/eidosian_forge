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
def is_body_allowed_for_status_code(status_code: Union[int, str, None]) -> bool:
    if status_code is None:
        return True
    if status_code in {'default', '1XX', '2XX', '3XX', '4XX', '5XX'}:
        return True
    current_status_code = int(status_code)
    return not (current_status_code < 200 or current_status_code in {204, 205, 304})