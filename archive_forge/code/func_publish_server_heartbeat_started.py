from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def publish_server_heartbeat_started(self, connection_id: _Address, awaited: bool) -> None:
    """Publish a ServerHeartbeatStartedEvent to all server heartbeat
        listeners.

        :Parameters:
         - `connection_id`: The address (host, port) pair of the connection.
         - `awaited`: True if this heartbeat is part of an awaitable hello command.
        """
    event = ServerHeartbeatStartedEvent(connection_id, awaited)
    for subscriber in self.__server_heartbeat_listeners:
        try:
            subscriber.started(event)
        except Exception:
            _handle_exception()