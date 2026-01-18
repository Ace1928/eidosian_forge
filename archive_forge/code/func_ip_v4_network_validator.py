from __future__ import annotations as _annotations
import math
import re
import typing
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from typing import Any
from pydantic_core import PydanticCustomError, core_schema
from pydantic_core._pydantic_core import PydanticKnownError
def ip_v4_network_validator(__input_value: Any) -> IPv4Network:
    """Assume IPv4Network initialised with a default `strict` argument.

    See more:
    https://docs.python.org/library/ipaddress.html#ipaddress.IPv4Network
    """
    if isinstance(__input_value, IPv4Network):
        return __input_value
    try:
        return IPv4Network(__input_value)
    except ValueError:
        raise PydanticCustomError('ip_v4_network', 'Input is not a valid IPv4 network')