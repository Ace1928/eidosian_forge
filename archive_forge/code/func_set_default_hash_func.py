from __future__ import annotations as _annotations
import operator
import typing
import warnings
import weakref
from abc import ABCMeta
from functools import partial
from types import FunctionType
from typing import Any, Callable, Generic
import typing_extensions
from pydantic_core import PydanticUndefined, SchemaSerializer
from typing_extensions import dataclass_transform, deprecated
from ..errors import PydanticUndefinedAnnotation, PydanticUserError
from ..plugin._schema_validator import create_schema_validator
from ..warnings import GenericBeforeBaseModelWarning, PydanticDeprecatedSince20
from ._config import ConfigWrapper
from ._decorators import DecoratorInfos, PydanticDescriptorProxy, get_attribute_from_bases
from ._fields import collect_model_fields, is_valid_field_name, is_valid_privateattr_name
from ._generate_schema import GenerateSchema
from ._generics import PydanticGenericMetadata, get_model_typevars_map
from ._mock_val_ser import MockValSer, set_model_mocks
from ._schema_generation_shared import CallbackGetCoreSchemaHandler
from ._signature import generate_pydantic_signature
from ._typing_extra import get_cls_types_namespace, is_annotated, is_classvar, parent_frame_namespace
from ._utils import ClassAttribute, SafeGetItemProxy
from ._validate_call import ValidateCallWrapper
def set_default_hash_func(cls: type[BaseModel], bases: tuple[type[Any], ...]) -> None:
    base_hash_func = get_attribute_from_bases(bases, '__hash__')
    new_hash_func = make_hash_func(cls)
    if base_hash_func in {None, object.__hash__} or getattr(base_hash_func, '__code__', None) == new_hash_func.__code__:
        cls.__hash__ = new_hash_func