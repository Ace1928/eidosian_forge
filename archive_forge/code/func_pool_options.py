from __future__ import annotations
import threading
import traceback
from typing import Any, Collection, Optional, Type, Union
from bson.objectid import ObjectId
from pymongo import common, monitor, pool
from pymongo.common import LOCAL_THRESHOLD_MS, SERVER_SELECTION_TIMEOUT
from pymongo.errors import ConfigurationError
from pymongo.pool import Pool, PoolOptions
from pymongo.server_description import ServerDescription
from pymongo.topology_description import TOPOLOGY_TYPE, _ServerSelector
@property
def pool_options(self) -> PoolOptions:
    return self._pool_options