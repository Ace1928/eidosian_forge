from __future__ import annotations as _annotations
import math
import re
import typing
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from typing import Any
from pydantic_core import PydanticCustomError, core_schema
from pydantic_core._pydantic_core import PydanticKnownError
def ip_v6_interface_validator(__input_value: Any) -> IPv6Interface:
    if isinstance(__input_value, IPv6Interface):
        return __input_value
    try:
        return IPv6Interface(__input_value)
    except ValueError:
        raise PydanticCustomError('ip_v6_interface', 'Input is not a valid IPv6 interface')