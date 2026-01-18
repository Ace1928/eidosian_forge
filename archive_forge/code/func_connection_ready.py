from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def connection_ready(self, event: ConnectionReadyEvent) -> None:
    """Abstract method to handle a :class:`ConnectionReadyEvent`.

        Emitted when a connection has finished its setup, and is now ready to
        use.

        :Parameters:
          - `event`: An instance of :class:`ConnectionReadyEvent`.
        """
    raise NotImplementedError