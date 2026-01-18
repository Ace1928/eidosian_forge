import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Optional, Type, TypeVar, Union
from .parse import Protocol, load_file, load_str_bytes
from .types import StrBytes
from .typing import display_as_type
def schema_json_of(type_: Any, *, title: Optional[NameFactory]=None, **schema_json_kwargs: Any) -> str:
    """Generate a JSON schema (as JSON) for the passed model or dynamically generated one"""
    return _get_parsing_type(type_, type_name=title).schema_json(**schema_json_kwargs)