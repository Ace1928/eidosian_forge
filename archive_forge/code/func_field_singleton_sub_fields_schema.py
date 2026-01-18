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
def field_singleton_sub_fields_schema(field: ModelField, *, by_alias: bool, model_name_map: Dict[TypeModelOrEnum, str], ref_template: str, schema_overrides: bool=False, ref_prefix: Optional[str]=None, known_models: TypeModelSet) -> Tuple[Dict[str, Any], Dict[str, Any], Set[str]]:
    """
    This function is indirectly used by ``field_schema()``, you probably should be using that function.

    Take a list of Pydantic ``ModelField`` from the declaration of a type with parameters, and generate their
    schema. I.e., fields used as "type parameters", like ``str`` and ``int`` in ``Tuple[str, int]``.
    """
    sub_fields = cast(List[ModelField], field.sub_fields)
    definitions = {}
    nested_models: Set[str] = set()
    if len(sub_fields) == 1:
        return field_type_schema(sub_fields[0], by_alias=by_alias, model_name_map=model_name_map, schema_overrides=schema_overrides, ref_prefix=ref_prefix, ref_template=ref_template, known_models=known_models)
    else:
        s: Dict[str, Any] = {}
        field_has_discriminator: bool = field.discriminator_key is not None
        if field_has_discriminator:
            assert field.sub_fields_mapping is not None
            discriminator_models_refs: Dict[str, Union[str, Dict[str, Any]]] = {}
            for discriminator_value, sub_field in field.sub_fields_mapping.items():
                if isinstance(discriminator_value, Enum):
                    discriminator_value = str(discriminator_value.value)
                if is_union(get_origin(sub_field.type_)):
                    sub_models = get_sub_types(sub_field.type_)
                    discriminator_models_refs[discriminator_value] = {model_name_map[sub_model]: get_schema_ref(model_name_map[sub_model], ref_prefix, ref_template, False) for sub_model in sub_models}
                else:
                    sub_field_type = sub_field.type_
                    if hasattr(sub_field_type, '__pydantic_model__'):
                        sub_field_type = sub_field_type.__pydantic_model__
                    discriminator_model_name = model_name_map[sub_field_type]
                    discriminator_model_ref = get_schema_ref(discriminator_model_name, ref_prefix, ref_template, False)
                    discriminator_models_refs[discriminator_value] = discriminator_model_ref['$ref']
            s['discriminator'] = {'propertyName': field.discriminator_alias, 'mapping': discriminator_models_refs}
        sub_field_schemas = []
        for sf in sub_fields:
            sub_schema, sub_definitions, sub_nested_models = field_type_schema(sf, by_alias=by_alias, model_name_map=model_name_map, schema_overrides=schema_overrides, ref_prefix=ref_prefix, ref_template=ref_template, known_models=known_models)
            definitions.update(sub_definitions)
            if schema_overrides and 'allOf' in sub_schema:
                sub_schema = sub_schema['allOf'][0]
            if sub_schema.keys() == {'discriminator', 'oneOf'}:
                sub_schema.pop('discriminator')
            sub_field_schemas.append(sub_schema)
            nested_models.update(sub_nested_models)
        s['oneOf' if field_has_discriminator else 'anyOf'] = sub_field_schemas
        return (s, definitions, nested_models)