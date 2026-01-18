import re
import warnings
from collections import defaultdict
from dataclasses import is_dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from pathlib import Path
from typing import (
from uuid import UUID
from typing_extensions import Annotated, Literal
from .fields import (
from .json import pydantic_encoder
from .networks import AnyUrl, EmailStr
from .types import (
from .typing import (
from .utils import ROOT_KEY, get_model, lenient_issubclass
def get_field_info_schema(field: ModelField, schema_overrides: bool=False) -> Tuple[Dict[str, Any], bool]:
    schema_: Dict[str, Any] = {}
    if field.field_info.title or not lenient_issubclass(field.type_, Enum):
        schema_['title'] = field.field_info.title or field.alias.title().replace('_', ' ')
    if field.field_info.title:
        schema_overrides = True
    if field.field_info.description:
        schema_['description'] = field.field_info.description
        schema_overrides = True
    if not field.required and field.default is not None and (not is_callable_type(field.outer_type_)):
        schema_['default'] = encode_default(field.default)
        schema_overrides = True
    return (schema_, schema_overrides)