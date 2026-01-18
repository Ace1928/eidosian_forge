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
def get_schema_ref(name: str, ref_prefix: Optional[str], ref_template: str, schema_overrides: bool) -> Dict[str, Any]:
    if ref_prefix:
        schema_ref = {'$ref': ref_prefix + name}
    else:
        schema_ref = {'$ref': ref_template.format(model=name)}
    return {'allOf': [schema_ref]} if schema_overrides else schema_ref