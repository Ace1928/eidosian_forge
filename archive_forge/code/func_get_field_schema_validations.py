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
def get_field_schema_validations(field: ModelField) -> Dict[str, Any]:
    """
    Get the JSON Schema validation keywords for a ``field`` with an annotation of
    a Pydantic ``FieldInfo`` with validation arguments.
    """
    f_schema: Dict[str, Any] = {}
    if lenient_issubclass(field.type_, Enum):
        if field.field_info.extra:
            f_schema.update(field.field_info.extra)
        return f_schema
    if lenient_issubclass(field.type_, (str, bytes)):
        for attr_name, t, keyword in _str_types_attrs:
            attr = getattr(field.field_info, attr_name, None)
            if isinstance(attr, t):
                f_schema[keyword] = attr
    if lenient_issubclass(field.type_, numeric_types) and (not issubclass(field.type_, bool)):
        for attr_name, t, keyword in _numeric_types_attrs:
            attr = getattr(field.field_info, attr_name, None)
            if isinstance(attr, t):
                f_schema[keyword] = attr
    if field.field_info is not None and field.field_info.const:
        f_schema['const'] = field.default
    if field.field_info.extra:
        f_schema.update(field.field_info.extra)
    modify_schema = getattr(field.outer_type_, '__modify_schema__', None)
    if modify_schema:
        _apply_modify_schema(modify_schema, field, f_schema)
    return f_schema