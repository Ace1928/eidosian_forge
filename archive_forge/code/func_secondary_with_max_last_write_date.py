from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, TypeVar, cast
from pymongo.server_type import SERVER_TYPE
def secondary_with_max_last_write_date(self) -> Optional[ServerDescription]:
    secondaries = secondary_server_selector(self)
    if secondaries.server_descriptions:
        return max(secondaries.server_descriptions, key=lambda sd: cast(float, sd.last_write_date))
    return None