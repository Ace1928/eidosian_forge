from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def publish_pool_ready(self, address: _Address) -> None:
    """Publish a :class:`PoolReadyEvent` to all pool listeners."""
    event = PoolReadyEvent(address)
    for subscriber in self.__cmap_listeners:
        try:
            subscriber.pool_ready(event)
        except Exception:
            _handle_exception()