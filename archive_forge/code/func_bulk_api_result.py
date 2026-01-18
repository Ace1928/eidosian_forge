from __future__ import annotations
from typing import Any, Mapping, Optional, cast
from pymongo.errors import InvalidOperation
@property
def bulk_api_result(self) -> dict[str, Any]:
    """The raw bulk API result."""
    return self.__bulk_api_result