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
def get_openapi_schema_by_role(user_role: Optional[Union['UserRole', str]]=None, request: Optional['Request']=None, app: Optional['FastAPI']=None, force_https: Optional[bool]=None) -> Dict[str, Any]:
    """
        Get the openapi schema by role
        """
    module_schemas = get_module_by_role_schema(module_name=module_name)
    role = user_role or UserRole.ANON
    role_spec = module_schemas[role]
    if not role_spec.openapi_schema:
        if verbose:
            logger.info('Generating OpenAPI Schema', prefix=role)
        schema = copy.deepcopy(app.openapi()) if app else copy.deepcopy(get_openapi_schema(module_name))
        patch_openapi_description(role_spec=role_spec, schema=schema, request=request, force_https=domain_name_force_https or force_https)
        patch_openapi_paths(role_spec=role_spec, schema=schema)
        if replace_patches:
            for patch in replace_patches:
                key, value = patch
                if callable(value):
                    value = value()
                schema = json_replace(schema=schema, key=key, value=value, key_start=replace_key_start, key_end=replace_key_end, sep_char=replace_sep_char)
        role_spec.openapi_schema = schema
        set_module_role_spec(module_name=module_name, role_spec=role_spec)
    return role_spec.openapi_schema