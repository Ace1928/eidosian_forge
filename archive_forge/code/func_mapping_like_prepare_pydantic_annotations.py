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
def mapping_like_prepare_pydantic_annotations(source_type: Any, annotations: Iterable[Any], _config: ConfigDict) -> tuple[Any, list[Any]] | None:
    origin: Any = get_origin(source_type)
    mapped_origin = MAPPING_ORIGIN_MAP.get(origin, None) if origin else MAPPING_ORIGIN_MAP.get(source_type, None)
    if mapped_origin is None:
        return None
    args = get_args(source_type)
    if not args:
        args = (Any, Any)
    elif mapped_origin is collections.Counter:
        if len(args) != 1:
            raise ValueError('Expected Counter to have exactly 1 generic parameter')
        args = (args[0], int)
    elif len(args) != 2:
        raise ValueError('Expected mapping to have exactly 2 generic parameters')
    keys_source_type, values_source_type = args
    metadata, remaining_annotations = _known_annotated_metadata.collect_known_metadata(annotations)
    _known_annotated_metadata.check_metadata(metadata, _known_annotated_metadata.SEQUENCE_CONSTRAINTS, source_type)
    return (source_type, [MappingValidator(mapped_origin, keys_source_type, values_source_type, **metadata), *remaining_annotations])