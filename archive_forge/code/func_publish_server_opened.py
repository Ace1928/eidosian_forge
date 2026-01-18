from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def publish_server_opened(self, server_address: _Address, topology_id: ObjectId) -> None:
    """Publish a ServerOpeningEvent to all server listeners.

        :Parameters:
         - `server_address`: The address (host, port) pair of the server.
         - `topology_id`: A unique identifier for the topology this server
           is a part of.
        """
    event = ServerOpeningEvent(server_address, topology_id)
    for subscriber in self.__server_listeners:
        try:
            subscriber.opened(event)
        except Exception:
            _handle_exception()