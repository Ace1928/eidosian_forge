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
def create_openapi_schema_patch(module_name: str, schemas_patches: Dict[str, Dict[str, str]], excluded_schemas: List[str], replace_patches: Optional[List[Tuple[str, Union[Callable, Optional[str]]]]]=None, replace_key_start: Optional[str]=KEY_START, replace_key_end: Optional[str]=KEY_END, replace_sep_char: Optional[str]=KEY_SEP, replace_domain_key: Optional[str]=DOMAIN_KEY) -> Callable[[Dict[str, Any]], Dict[str, Any]]:
    """
    Create an openapi schema patch
    """

    def patch_openapi_schema_wrapper(openapi_schema: Dict[str, Any], overwrite: Optional[bool]=None, verbose: Optional[bool]=True, **kwargs) -> Dict[str, Any]:
        """
        Patch the openapi schema wrapper
        """
        return patch_openapi_schema(openapi_schema=openapi_schema, schemas_patches=schemas_patches, excluded_schemas=excluded_schemas, module_name=module_name, overwrite=overwrite, verbose=verbose, replace_patches=replace_patches, replace_key_start=replace_key_start, replace_key_end=replace_key_end, replace_sep_char=replace_sep_char)
    return patch_openapi_schema_wrapper