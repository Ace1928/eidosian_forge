from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def publish_command_success(self, duration: timedelta, reply: _DocumentOut, command_name: str, request_id: int, connection_id: _Address, op_id: Optional[int]=None, service_id: Optional[ObjectId]=None, speculative_hello: bool=False, database_name: str='') -> None:
    """Publish a CommandSucceededEvent to all command listeners.

        :Parameters:
          - `duration`: The command duration as a datetime.timedelta.
          - `reply`: The server reply document.
          - `command_name`: The command name.
          - `request_id`: The request id for this operation.
          - `connection_id`: The address (host, port) of the server this
            command was sent to.
          - `op_id`: The (optional) operation id for this operation.
          - `service_id`: The service_id this command was sent to, or ``None``.
          - `speculative_hello`: Was the command sent with speculative auth?
          - `database_name`: The database this command was sent to, or ``""``.
        """
    if op_id is None:
        op_id = request_id
    if speculative_hello:
        reply = {}
    event = CommandSucceededEvent(duration, reply, command_name, request_id, connection_id, op_id, service_id, database_name=database_name)
    for subscriber in self.__command_listeners:
        try:
            subscriber.succeeded(event)
        except Exception:
            _handle_exception()