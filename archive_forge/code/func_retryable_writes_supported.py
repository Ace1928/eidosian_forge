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
def retryable_writes_supported(self) -> bool:
    """Checks if this server supports retryable writes."""
    return self._ls_timeout_minutes is not None and self._server_type in (SERVER_TYPE.Mongos, SERVER_TYPE.RSPrimary) or self._server_type == SERVER_TYPE.LoadBalancer