import re
import warnings
from dataclasses import is_dataclass
from typing import (
from weakref import WeakKeyDictionary
import fastapi
from fastapi._compat import (
from fastapi.datastructures import DefaultPlaceholder, DefaultType
from pydantic import BaseModel, create_model
from pydantic.fields import FieldInfo
from typing_extensions import Literal
def create_cloned_field(field: ModelField, *, cloned_types: Optional[MutableMapping[Type[BaseModel], Type[BaseModel]]]=None) -> ModelField:
    if PYDANTIC_V2:
        return field
    if cloned_types is None:
        cloned_types = _CLONED_TYPES_CACHE
    original_type = field.type_
    if is_dataclass(original_type) and hasattr(original_type, '__pydantic_model__'):
        original_type = original_type.__pydantic_model__
    use_type = original_type
    if lenient_issubclass(original_type, BaseModel):
        original_type = cast(Type[BaseModel], original_type)
        use_type = cloned_types.get(original_type)
        if use_type is None:
            use_type = create_model(original_type.__name__, __base__=original_type)
            cloned_types[original_type] = use_type
            for f in original_type.__fields__.values():
                use_type.__fields__[f.name] = create_cloned_field(f, cloned_types=cloned_types)
    new_field = create_response_field(name=field.name, type_=use_type)
    new_field.has_alias = field.has_alias
    new_field.alias = field.alias
    new_field.class_validators = field.class_validators
    new_field.default = field.default
    new_field.required = field.required
    new_field.model_config = field.model_config
    new_field.field_info = field.field_info
    new_field.allow_none = field.allow_none
    new_field.validate_always = field.validate_always
    if field.sub_fields:
        new_field.sub_fields = [create_cloned_field(sub_field, cloned_types=cloned_types) for sub_field in field.sub_fields]
    if field.key_field:
        new_field.key_field = create_cloned_field(field.key_field, cloned_types=cloned_types)
    new_field.validators = field.validators
    new_field.pre_validators = field.pre_validators
    new_field.post_validators = field.post_validators
    new_field.parse_json = field.parse_json
    new_field.shape = field.shape
    new_field.populate_validators()
    return new_field