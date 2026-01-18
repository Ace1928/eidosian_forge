from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def publish_connection_checked_out(self, address: _Address, connection_id: int) -> None:
    """Publish a :class:`ConnectionCheckedOutEvent` to all connection
        listeners.
        """
    event = ConnectionCheckedOutEvent(address, connection_id)
    for subscriber in self.__cmap_listeners:
        try:
            subscriber.connection_checked_out(event)
        except Exception:
            _handle_exception()