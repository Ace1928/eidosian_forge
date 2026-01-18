from __future__ import annotations
import os
from collections import defaultdict
from typing import (
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
def can_be_inlined(s: core_schema.DefinitionReferenceSchema, ref: str) -> bool:
    if ref_counts[ref] > 1:
        return False
    if involved_in_recursion.get(ref, False):
        return False
    if 'serialization' in s:
        return False
    if 'metadata' in s:
        metadata = s['metadata']
        for k in ('pydantic_js_functions', 'pydantic_js_annotation_functions', 'pydantic.internal.union_discriminator'):
            if k in metadata:
                return False
    return True