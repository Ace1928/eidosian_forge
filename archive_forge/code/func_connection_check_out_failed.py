from __future__ import annotations
import datetime
from collections import abc, namedtuple
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence
from bson.objectid import ObjectId
from pymongo.hello import Hello, HelloCompat
from pymongo.helpers import _handle_exception
from pymongo.typings import _Address, _DocumentOut
def connection_check_out_failed(self, event: ConnectionCheckOutFailedEvent) -> None:
    """Abstract method to handle a :class:`ConnectionCheckOutFailedEvent`.

        Emitted when the driver's attempt to check out a connection fails.

        :Parameters:
          - `event`: An instance of :class:`ConnectionCheckOutFailedEvent`.
        """
    raise NotImplementedError