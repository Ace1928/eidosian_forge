from __future__ import annotations as _annotations
import math
import re
import typing
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from typing import Any
from pydantic_core import PydanticCustomError, core_schema
from pydantic_core._pydantic_core import PydanticKnownError
def sequence_validator(__input_value: typing.Sequence[Any], validator: core_schema.ValidatorFunctionWrapHandler) -> typing.Sequence[Any]:
    """Validator for `Sequence` types, isinstance(v, Sequence) has already been called."""
    value_type = type(__input_value)
    if issubclass(value_type, (str, bytes)):
        raise PydanticCustomError('sequence_str', "'{type_name}' instances are not allowed as a Sequence value", {'type_name': value_type.__name__})
    v_list = validator(__input_value)
    if value_type == list:
        return v_list
    elif issubclass(value_type, range):
        return v_list
    else:
        return value_type(v_list)