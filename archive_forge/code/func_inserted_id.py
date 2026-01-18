from __future__ import annotations
from typing import Any, Mapping, Optional, cast
from pymongo.errors import InvalidOperation
@property
def inserted_id(self) -> Any:
    """The inserted document's _id."""
    return self.__inserted_id