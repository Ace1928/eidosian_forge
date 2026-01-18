from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def publish_connection_closed(self, address: _Address, connection_id: int, reason: str) -> None:
    """Publish a :class:`ConnectionClosedEvent` to all connection
        listeners.
        """
    event = ConnectionClosedEvent(address, connection_id, reason)
    for subscriber in self.__cmap_listeners:
        try:
            subscriber.connection_closed(event)
        except Exception:
            _handle_exception()