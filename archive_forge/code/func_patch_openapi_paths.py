from __future__ import annotations
import re
import json
import copy
import contextlib
import operator
from abc import ABC
from pydantic import BaseModel
from typing import Any, Dict, List, Optional, Union, Callable, Tuple, TYPE_CHECKING
from lazyops.utils import logger
from lazyops.utils.lazy import lazy_import
from lazyops.libs.fastapi_utils.types.user_roles import UserRole
def patch_openapi_paths(role_spec: OpenAPIRoleSpec, schema: Dict[str, Union[Dict[str, Union[Dict[str, Any], Any]], Any]]) -> Dict[str, Any]:
    """
        Patch the openapi schema paths
        """
    for path, methods in schema['paths'].items():
        for method, spec in methods.items():
            if 'tags' in spec and any((tag in role_spec.excluded_tags for tag in spec['tags'])):
                role_spec.excluded_paths.append(path)
    for path in role_spec.excluded_paths:
        if isinstance(path, str):
            schema['paths'].pop(path, None)
        elif isinstance(path, dict):
            schema['paths'][path['path']].pop(path['method'], None)
    for path in default_exclude_paths:
        if path in role_spec.included_paths:
            continue
        if isinstance(path, str):
            schema['paths'].pop(path, None)
        elif isinstance(path, dict):
            schema['paths'][path['path']].pop(path['method'], None)
    if 'components' not in schema:
        return schema
    if 'schemas' not in schema['components']:
        return schema
    _schemas_to_remove = []
    for schema_name in role_spec.excluded_schemas:
        if '*' in schema_name:
            _schemas_to_remove += [schema for schema in schema['components']['schemas'].keys() if re.match(schema_name, schema)]
        else:
            _schemas_to_remove.append(schema_name)
    _schemas_to_remove = list(set(_schemas_to_remove))
    for schema_name in _schemas_to_remove:
        schema['components']['schemas'].pop(schema_name, None)
    if role_spec.extra_schemas:
        role_spec.populate_extra_schemas()
        schema['components']['schemas'].update(role_spec.extra_schemas_data)
    schema['components']['schemas'] = dict(sorted(schema['components']['schemas'].items(), key=lambda x: operator.itemgetter('title')(x[1])))
    return schema