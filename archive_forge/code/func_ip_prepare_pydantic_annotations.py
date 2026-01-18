from __future__ import annotations as _annotations
import collections
import collections.abc
import dataclasses
import decimal
import inspect
import os
import typing
from enum import Enum
from functools import partial
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from typing import Any, Callable, Iterable, TypeVar
import typing_extensions
from pydantic_core import (
from typing_extensions import get_args, get_origin
from pydantic.errors import PydanticSchemaGenerationError
from pydantic.fields import FieldInfo
from pydantic.types import Strict
from ..config import ConfigDict
from ..json_schema import JsonSchemaValue, update_json_schema
from . import _known_annotated_metadata, _typing_extra, _validators
from ._core_utils import get_type_ref
from ._internal_dataclass import slots_true
from ._schema_generation_shared import GetCoreSchemaHandler, GetJsonSchemaHandler
def ip_prepare_pydantic_annotations(source_type: Any, annotations: Iterable[Any], _config: ConfigDict) -> tuple[Any, list[Any]] | None:

    def make_strict_ip_schema(tp: type[Any]) -> CoreSchema:
        return core_schema.json_or_python_schema(json_schema=core_schema.no_info_after_validator_function(tp, core_schema.str_schema()), python_schema=core_schema.is_instance_schema(tp))
    if source_type is IPv4Address:
        return (source_type, [SchemaTransformer(lambda _1, _2: core_schema.lax_or_strict_schema(lax_schema=core_schema.no_info_plain_validator_function(_validators.ip_v4_address_validator), strict_schema=make_strict_ip_schema(IPv4Address), serialization=core_schema.to_string_ser_schema()), lambda _1, _2: {'type': 'string', 'format': 'ipv4'}), *annotations])
    if source_type is IPv4Network:
        return (source_type, [SchemaTransformer(lambda _1, _2: core_schema.lax_or_strict_schema(lax_schema=core_schema.no_info_plain_validator_function(_validators.ip_v4_network_validator), strict_schema=make_strict_ip_schema(IPv4Network), serialization=core_schema.to_string_ser_schema()), lambda _1, _2: {'type': 'string', 'format': 'ipv4network'}), *annotations])
    if source_type is IPv4Interface:
        return (source_type, [SchemaTransformer(lambda _1, _2: core_schema.lax_or_strict_schema(lax_schema=core_schema.no_info_plain_validator_function(_validators.ip_v4_interface_validator), strict_schema=make_strict_ip_schema(IPv4Interface), serialization=core_schema.to_string_ser_schema()), lambda _1, _2: {'type': 'string', 'format': 'ipv4interface'}), *annotations])
    if source_type is IPv6Address:
        return (source_type, [SchemaTransformer(lambda _1, _2: core_schema.lax_or_strict_schema(lax_schema=core_schema.no_info_plain_validator_function(_validators.ip_v6_address_validator), strict_schema=make_strict_ip_schema(IPv6Address), serialization=core_schema.to_string_ser_schema()), lambda _1, _2: {'type': 'string', 'format': 'ipv6'}), *annotations])
    if source_type is IPv6Network:
        return (source_type, [SchemaTransformer(lambda _1, _2: core_schema.lax_or_strict_schema(lax_schema=core_schema.no_info_plain_validator_function(_validators.ip_v6_network_validator), strict_schema=make_strict_ip_schema(IPv6Network), serialization=core_schema.to_string_ser_schema()), lambda _1, _2: {'type': 'string', 'format': 'ipv6network'}), *annotations])
    if source_type is IPv6Interface:
        return (source_type, [SchemaTransformer(lambda _1, _2: core_schema.lax_or_strict_schema(lax_schema=core_schema.no_info_plain_validator_function(_validators.ip_v6_interface_validator), strict_schema=make_strict_ip_schema(IPv6Interface), serialization=core_schema.to_string_ser_schema()), lambda _1, _2: {'type': 'string', 'format': 'ipv6interface'}), *annotations])
    return None