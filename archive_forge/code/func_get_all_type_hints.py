import sys
import typing
from collections.abc import Callable
from os import PathLike
from typing import (  # type: ignore
from typing_extensions import (
def get_all_type_hints(obj: Any, globalns: Any=None, localns: Any=None) -> Any:
    return get_type_hints(obj, globalns, localns, include_extras=True)