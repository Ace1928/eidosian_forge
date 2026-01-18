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
def get_module_by_role_schema(module_name: str) -> Dict[UserRole, OpenAPIRoleSpec]:
    """
    Get the module by role schema
    """
    global _openapi_schemas_by_role
    return _openapi_schemas_by_role[module_name]