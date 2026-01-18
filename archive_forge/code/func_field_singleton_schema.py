import re
import warnings
from collections import defaultdict
from dataclasses import is_dataclass
from datetime import date, datetime, time, timedelta
from decimal import Decimal
from enum import Enum
from ipaddress import IPv4Address, IPv4Interface, IPv4Network, IPv6Address, IPv6Interface, IPv6Network
from pathlib import Path
from typing import (
from uuid import UUID
from typing_extensions import Annotated, Literal
from .fields import (
from .json import pydantic_encoder
from .networks import AnyUrl, EmailStr
from .types import (
from .typing import (
from .utils import ROOT_KEY, get_model, lenient_issubclass
def field_singleton_schema(field: ModelField, *, by_alias: bool, model_name_map: Dict[TypeModelOrEnum, str], ref_template: str, schema_overrides: bool=False, ref_prefix: Optional[str]=None, known_models: TypeModelSet) -> Tuple[Dict[str, Any], Dict[str, Any], Set[str]]:
    """
    This function is indirectly used by ``field_schema()``, you should probably be using that function.

    Take a single Pydantic ``ModelField``, and return its schema and any additional definitions from sub-models.
    """
    from .main import BaseModel
    definitions: Dict[str, Any] = {}
    nested_models: Set[str] = set()
    field_type = field.type_
    if field.sub_fields and (field.field_info and field.field_info.const or not lenient_issubclass(field_type, BaseModel)):
        return field_singleton_sub_fields_schema(field, by_alias=by_alias, model_name_map=model_name_map, schema_overrides=schema_overrides, ref_prefix=ref_prefix, ref_template=ref_template, known_models=known_models)
    if field_type is Any or field_type is object or field_type.__class__ == TypeVar or (get_origin(field_type) is type):
        return ({}, definitions, nested_models)
    if is_none_type(field_type):
        return ({'type': 'null'}, definitions, nested_models)
    if is_callable_type(field_type):
        raise SkipField(f'Callable {field.name} was excluded from schema since JSON schema has no equivalent type.')
    f_schema: Dict[str, Any] = {}
    if field.field_info is not None and field.field_info.const:
        f_schema['const'] = field.default
    if is_literal_type(field_type):
        values = tuple((x.value if isinstance(x, Enum) else x for x in all_literal_values(field_type)))
        if len({v.__class__ for v in values}) > 1:
            return field_schema(multitypes_literal_field_for_schema(values, field), by_alias=by_alias, model_name_map=model_name_map, ref_prefix=ref_prefix, ref_template=ref_template, known_models=known_models)
        field_type = values[0].__class__
        f_schema['enum'] = list(values)
        add_field_type_to_schema(field_type, f_schema)
    elif lenient_issubclass(field_type, Enum):
        enum_name = model_name_map[field_type]
        f_schema, schema_overrides = get_field_info_schema(field, schema_overrides)
        f_schema.update(get_schema_ref(enum_name, ref_prefix, ref_template, schema_overrides))
        definitions[enum_name] = enum_process_schema(field_type, field=field)
    elif is_namedtuple(field_type):
        sub_schema, *_ = model_process_schema(field_type.__pydantic_model__, by_alias=by_alias, model_name_map=model_name_map, ref_prefix=ref_prefix, ref_template=ref_template, known_models=known_models, field=field)
        items_schemas = list(sub_schema['properties'].values())
        f_schema.update({'type': 'array', 'items': items_schemas, 'minItems': len(items_schemas), 'maxItems': len(items_schemas)})
    elif not hasattr(field_type, '__pydantic_model__'):
        add_field_type_to_schema(field_type, f_schema)
        modify_schema = getattr(field_type, '__modify_schema__', None)
        if modify_schema:
            _apply_modify_schema(modify_schema, field, f_schema)
    if f_schema:
        return (f_schema, definitions, nested_models)
    if lenient_issubclass(getattr(field_type, '__pydantic_model__', None), BaseModel):
        field_type = field_type.__pydantic_model__
    if issubclass(field_type, BaseModel):
        model_name = model_name_map[field_type]
        if field_type not in known_models:
            sub_schema, sub_definitions, sub_nested_models = model_process_schema(field_type, by_alias=by_alias, model_name_map=model_name_map, ref_prefix=ref_prefix, ref_template=ref_template, known_models=known_models, field=field)
            definitions.update(sub_definitions)
            definitions[model_name] = sub_schema
            nested_models.update(sub_nested_models)
        else:
            nested_models.add(model_name)
        schema_ref = get_schema_ref(model_name, ref_prefix, ref_template, schema_overrides)
        return (schema_ref, definitions, nested_models)
    args = get_args(field_type)
    if args is not None and (not args) and (Generic in field_type.__bases__):
        return (f_schema, definitions, nested_models)
    raise ValueError(f'Value not declarable with JSON Schema, field: {field}')