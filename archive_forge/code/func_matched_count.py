from __future__ import annotations
from typing import Any, Mapping, Optional, cast
from pymongo.errors import InvalidOperation
@property
def matched_count(self) -> int:
    """The number of documents matched for an update."""
    self._raise_if_unacknowledged('matched_count')
    return cast(int, self.__bulk_api_result.get('nMatched'))