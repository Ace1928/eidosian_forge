from __future__ import annotations
import time
import warnings
from typing import Any, Mapping, Optional
from bson import EPOCH_NAIVE
from bson.objectid import ObjectId
from pymongo.hello import Hello
from pymongo.server_type import SERVER_TYPE
from pymongo.typings import ClusterTime, _Address
@property
def all_hosts(self) -> set[tuple[str, int]]:
    """List of hosts, passives, and arbiters known to this server."""
    return self._all_hosts