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
def patch_openapi_description(role_spec: OpenAPIRoleSpec, schema: Dict[str, Union[Dict[str, Union[Dict[str, Any], Any]], Any]], request: Optional['Request']=None, force_https: Optional[bool]=None) -> Dict[str, Any]:
    """
        Patch the openapi schema description
        """
    nonlocal domain_name
    if domain_name is None:
        domain_name = get_server_domain(request=request, module_name=module_name, module_domains=module_domains, verbose=verbose, force_https=domain_name_force_https or force_https)
    if domain_name:
        replace_domain_start = f'<<{replace_domain_key}>>'
        replace_domain_end = f'>>{replace_domain_key}<<'
        schema['info']['description'] = schema['info']['description'].replace(replace_domain_start, domain_name).replace(replace_domain_end, domain_name)
        if role_spec.has_description_callable:
            schema['info']['description'] = role_spec.description_callable(schema['info']['description'], domain_name, role_spec)
    return schema