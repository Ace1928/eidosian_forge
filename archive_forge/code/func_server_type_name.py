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
def server_type_name(self) -> str:
    """The server type as a human readable string.

        .. versionadded:: 3.4
        """
    return SERVER_TYPE._fields[self._server_type]