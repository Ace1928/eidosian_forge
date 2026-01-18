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
def replica_set_name(self) -> Optional[str]:
    """Replica set name or None."""
    return self._replica_set_name