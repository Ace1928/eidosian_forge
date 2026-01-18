import sys
import typing
from collections.abc import Callable
from os import PathLike
from typing import (  # type: ignore
from typing_extensions import (
def is_typeddict_special(type_: Any) -> bool:
    """
    Check if type is a TypedDict special form (Required or NotRequired).
    """
    return _check_typeddict_special(type_) or _check_typeddict_special(get_origin(type_))