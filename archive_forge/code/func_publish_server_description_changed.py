from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def publish_server_description_changed(self, previous_description: ServerDescription, new_description: ServerDescription, server_address: _Address, topology_id: ObjectId) -> None:
    """Publish a ServerDescriptionChangedEvent to all server listeners.

        :Parameters:
         - `previous_description`: The previous server description.
         - `server_address`: The address (host, port) pair of the server.
         - `new_description`: The new server description.
         - `topology_id`: A unique identifier for the topology this server
           is a part of.
        """
    event = ServerDescriptionChangedEvent(previous_description, new_description, server_address, topology_id)
    for subscriber in self.__server_listeners:
        try:
            subscriber.description_changed(event)
        except Exception:
            _handle_exception()