from __future__ import annotations
import os
from collections import defaultdict
from typing import (
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
def handle_arguments_schema(self, schema: core_schema.ArgumentsSchema, f: Walk) -> core_schema.CoreSchema:
    replaced_arguments_schema: list[core_schema.ArgumentsParameter] = []
    for param in schema['arguments_schema']:
        replaced_param = param.copy()
        replaced_param['schema'] = self.walk(param['schema'], f)
        replaced_arguments_schema.append(replaced_param)
    schema['arguments_schema'] = replaced_arguments_schema
    if 'var_args_schema' in schema:
        schema['var_args_schema'] = self.walk(schema['var_args_schema'], f)
    if 'var_kwargs_schema' in schema:
        schema['var_kwargs_schema'] = self.walk(schema['var_kwargs_schema'], f)
    return schema