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
def enum_process_schema(enum: Type[Enum], *, field: Optional[ModelField]=None) -> Dict[str, Any]:
    """
    Take a single `enum` and generate its schema.

    This is similar to the `model_process_schema` function, but applies to ``Enum`` objects.
    """
    import inspect
    schema_: Dict[str, Any] = {'title': enum.__name__, 'description': inspect.cleandoc(enum.__doc__ or 'An enumeration.'), 'enum': [item.value for item in cast(Iterable[Enum], enum)]}
    add_field_type_to_schema(enum, schema_)
    modify_schema = getattr(enum, '__modify_schema__', None)
    if modify_schema:
        _apply_modify_schema(modify_schema, field, schema_)
    return schema_