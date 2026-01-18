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
def field_schema(field: ModelField, *, by_alias: bool=True, model_name_map: Dict[TypeModelOrEnum, str], ref_prefix: Optional[str]=None, ref_template: str=default_ref_template, known_models: Optional[TypeModelSet]=None) -> Tuple[Dict[str, Any], Dict[str, Any], Set[str]]:
    """
    Process a Pydantic field and return a tuple with a JSON Schema for it as the first item.
    Also return a dictionary of definitions with models as keys and their schemas as values. If the passed field
    is a model and has sub-models, and those sub-models don't have overrides (as ``title``, ``default``, etc), they
    will be included in the definitions and referenced in the schema instead of included recursively.

    :param field: a Pydantic ``ModelField``
    :param by_alias: use the defined alias (if any) in the returned schema
    :param model_name_map: used to generate the JSON Schema references to other models included in the definitions
    :param ref_prefix: the JSON Pointer prefix to use for references to other schemas, if None, the default of
      #/definitions/ will be used
    :param ref_template: Use a ``string.format()`` template for ``$ref`` instead of a prefix. This can be useful for
      references that cannot be represented by ``ref_prefix`` such as a definition stored in another file. For a
      sibling json file in a ``/schemas`` directory use ``"/schemas/${model}.json#"``.
    :param known_models: used to solve circular references
    :return: tuple of the schema for this field and additional definitions
    """
    s, schema_overrides = get_field_info_schema(field)
    validation_schema = get_field_schema_validations(field)
    if validation_schema:
        s.update(validation_schema)
        schema_overrides = True
    f_schema, f_definitions, f_nested_models = field_type_schema(field, by_alias=by_alias, model_name_map=model_name_map, schema_overrides=schema_overrides, ref_prefix=ref_prefix, ref_template=ref_template, known_models=known_models or set())
    if '$ref' in f_schema:
        return (f_schema, f_definitions, f_nested_models)
    else:
        s.update(f_schema)
        return (s, f_definitions, f_nested_models)