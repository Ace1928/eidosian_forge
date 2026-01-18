from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def publish_connection_check_out_failed(self, address: _Address, reason: str) -> None:
    """Publish a :class:`ConnectionCheckOutFailedEvent` to all connection
        listeners.
        """
    event = ConnectionCheckOutFailedEvent(address, reason)
    for subscriber in self.__cmap_listeners:
        try:
            subscriber.connection_check_out_failed(event)
        except Exception:
            _handle_exception()