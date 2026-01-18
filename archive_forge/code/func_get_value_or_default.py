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
def get_value_or_default(first_item: Union[DefaultPlaceholder, DefaultType], *extra_items: Union[DefaultPlaceholder, DefaultType]) -> Union[DefaultPlaceholder, DefaultType]:
    """
    Pass items or `DefaultPlaceholder`s by descending priority.

    The first one to _not_ be a `DefaultPlaceholder` will be returned.

    Otherwise, the first item (a `DefaultPlaceholder`) will be returned.
    """
    items = (first_item,) + extra_items
    for item in items:
        if not isinstance(item, DefaultPlaceholder):
            return item
    return first_item