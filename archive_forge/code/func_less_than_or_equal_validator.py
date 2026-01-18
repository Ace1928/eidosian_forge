from __future__ import annotations as _annotations
import math
import re
import typing
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from typing import Any
from pydantic_core import PydanticCustomError, core_schema
from pydantic_core._pydantic_core import PydanticKnownError
def less_than_or_equal_validator(x: Any, le: Any) -> Any:
    if not x <= le:
        raise PydanticKnownError('less_than_equal', {'le': le})
    return x