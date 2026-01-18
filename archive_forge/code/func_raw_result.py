from __future__ import annotations
from typing import Any, Mapping, Optional, cast
from pymongo.errors import InvalidOperation
@property
def raw_result(self) -> Mapping[str, Any]:
    """The raw result document returned by the server."""
    return self.__raw_result