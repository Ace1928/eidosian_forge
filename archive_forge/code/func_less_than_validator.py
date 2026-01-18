from __future__ import annotations as _annotations
import math
import re
import typing
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from typing import Any
from pydantic_core import PydanticCustomError, core_schema
from pydantic_core._pydantic_core import PydanticKnownError
def less_than_validator(x: Any, lt: Any) -> Any:
    if not x < lt:
        raise PydanticKnownError('less_than', {'lt': lt})
    return x