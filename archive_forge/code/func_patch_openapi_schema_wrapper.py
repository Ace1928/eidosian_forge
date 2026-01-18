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
def patch_openapi_schema_wrapper(openapi_schema: Dict[str, Any], overwrite: Optional[bool]=None, verbose: Optional[bool]=True, **kwargs) -> Dict[str, Any]:
    """
        Patch the openapi schema wrapper
        """
    return patch_openapi_schema(openapi_schema=openapi_schema, schemas_patches=schemas_patches, excluded_schemas=excluded_schemas, module_name=module_name, overwrite=overwrite, verbose=verbose, replace_patches=replace_patches, replace_key_start=replace_key_start, replace_key_end=replace_key_end, replace_sep_char=replace_sep_char)