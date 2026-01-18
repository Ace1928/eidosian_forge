from __future__ import annotations
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, TypeVar, cast
from pymongo.server_type import SERVER_TYPE
@property
def heartbeat_frequency(self) -> int:
    return self.topology_description.heartbeat_frequency