from __future__ import annotations
from typing import Any, Mapping, Optional, cast
from pymongo.errors import InvalidOperation
@property
def deleted_count(self) -> int:
    """The number of documents deleted."""
    self._raise_if_unacknowledged('deleted_count')
    return cast(int, self.__bulk_api_result.get('nRemoved'))