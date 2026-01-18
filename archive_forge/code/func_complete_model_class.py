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
def complete_model_class(cls: type[BaseModel], cls_name: str, config_wrapper: ConfigWrapper, *, raise_errors: bool=True, types_namespace: dict[str, Any] | None, create_model_module: str | None=None) -> bool:
    """Finish building a model class.

    This logic must be called after class has been created since validation functions must be bound
    and `get_type_hints` requires a class object.

    Args:
        cls: BaseModel or dataclass.
        cls_name: The model or dataclass name.
        config_wrapper: The config wrapper instance.
        raise_errors: Whether to raise errors.
        types_namespace: Optional extra namespace to look for types in.
        create_model_module: The module of the class to be created, if created by `create_model`.

    Returns:
        `True` if the model is successfully completed, else `False`.

    Raises:
        PydanticUndefinedAnnotation: If `PydanticUndefinedAnnotation` occurs in`__get_pydantic_core_schema__`
            and `raise_errors=True`.
    """
    typevars_map = get_model_typevars_map(cls)
    gen_schema = GenerateSchema(config_wrapper, types_namespace, typevars_map)
    handler = CallbackGetCoreSchemaHandler(partial(gen_schema.generate_schema, from_dunder_get_core_schema=False), gen_schema, ref_mode='unpack')
    if config_wrapper.defer_build:
        set_model_mocks(cls, cls_name)
        return False
    try:
        schema = cls.__get_pydantic_core_schema__(cls, handler)
    except PydanticUndefinedAnnotation as e:
        if raise_errors:
            raise
        set_model_mocks(cls, cls_name, f'`{e.name}`')
        return False
    core_config = config_wrapper.core_config(cls)
    try:
        schema = gen_schema.clean_schema(schema)
    except gen_schema.CollectedInvalid:
        set_model_mocks(cls, cls_name)
        return False
    cls.__pydantic_core_schema__ = schema
    cls.__pydantic_validator__ = create_schema_validator(schema, cls, create_model_module or cls.__module__, cls.__qualname__, 'create_model' if create_model_module else 'BaseModel', core_config, config_wrapper.plugin_settings)
    cls.__pydantic_serializer__ = SchemaSerializer(schema, core_config)
    cls.__pydantic_complete__ = True
    cls.__signature__ = ClassAttribute('__signature__', generate_pydantic_signature(init=cls.__init__, fields=cls.model_fields, config_wrapper=config_wrapper))
    return True