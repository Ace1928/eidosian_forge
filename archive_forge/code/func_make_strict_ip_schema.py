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
def make_strict_ip_schema(tp: type[Any]) -> CoreSchema:
    return core_schema.json_or_python_schema(json_schema=core_schema.no_info_after_validator_function(tp, core_schema.str_schema()), python_schema=core_schema.is_instance_schema(tp))