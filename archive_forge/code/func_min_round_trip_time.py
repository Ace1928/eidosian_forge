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
def min_round_trip_time(self) -> float:
    """The min latency from the most recent samples."""
    return self._min_round_trip_time