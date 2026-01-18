import contextlib
import datetime
import functools
import importlib
import warnings
from importlib.metadata import version
from typing import Any, Callable, Dict, Optional, Set, Tuple, Union
from packaging.version import parse
from requests import HTTPError, Response
from langchain_core.pydantic_v1 import SecretStr
def get_pydantic_field_names(pydantic_cls: Any) -> Set[str]:
    """Get field names, including aliases, for a pydantic class.

    Args:
        pydantic_cls: Pydantic class."""
    all_required_field_names = set()
    for field in pydantic_cls.__fields__.values():
        all_required_field_names.add(field.name)
        if field.has_alias:
            all_required_field_names.add(field.alias)
    return all_required_field_names