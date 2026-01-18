from __future__ import annotations
import dataclasses
import enum
import functools
from typing import IO, TYPE_CHECKING, Any, Optional, Set, Type, TypeVar, Union
from typing_extensions import get_args, get_origin
from .. import _fields, _resolver
def representer(dumper: DataclassDumper, data: Any) -> yaml.Node:
    if dataclasses.is_dataclass(data):
        return dumper.represent_mapping(tag=DATACLASS_YAML_TAG_PREFIX + name, mapping={field.name: getattr(data, field.name) for field in dataclasses.fields(data) if field.init})
    elif isinstance(data, enum.Enum):
        return dumper.represent_scalar(tag=ENUM_YAML_TAG_PREFIX + name, value=data.name)
    assert False