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
def set_module_role_spec(module_name: str, role_spec: OpenAPIRoleSpec):
    """
    Set the module role spec
    """
    global _openapi_schemas_by_role
    _openapi_schemas_by_role[module_name][role_spec.role] = role_spec