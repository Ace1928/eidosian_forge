from __future__ import annotations
import os
from collections import defaultdict
from typing import (
from pydantic_core import CoreSchema, core_schema
from pydantic_core import validate_core_schema as _validate_core_schema
from typing_extensions import TypeAliasType, TypeGuard, get_args, get_origin
from . import _repr
from ._typing_extra import is_generic_alias
def pretty_print_core_schema(schema: CoreSchema, include_metadata: bool=False) -> None:
    """Pretty print a CoreSchema using rich.
    This is intended for debugging purposes.

    Args:
        schema: The CoreSchema to print.
        include_metadata: Whether to include metadata in the output. Defaults to `False`.
    """
    from rich import print
    if not include_metadata:
        schema = _strip_metadata(schema)
    return print(schema)