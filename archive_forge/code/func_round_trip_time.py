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
def round_trip_time(self) -> Optional[float]:
    """The current average latency or None."""
    if self._address in self._host_to_round_trip_time:
        return self._host_to_round_trip_time[self._address]
    return self._round_trip_time