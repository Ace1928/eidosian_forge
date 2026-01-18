from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def connection_check_out_started(self, event: ConnectionCheckOutStartedEvent) -> None:
    """Abstract method to handle a :class:`ConnectionCheckOutStartedEvent`.

        Emitted when the driver starts attempting to check out a connection.

        :Parameters:
          - `event`: An instance of :class:`ConnectionCheckOutStartedEvent`.
        """
    raise NotImplementedError