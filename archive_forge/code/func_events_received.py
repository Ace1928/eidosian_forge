from __future__ import annotations
import enum
import logging
import uuid
from typing import Generator, List, Optional, Type, Union
from .exceptions import (
from .extensions import Extension
from .frames import (
from .http11 import Request, Response
from .streams import StreamReader
from .typing import LoggerLike, Origin, Subprotocol
def events_received(self) -> List[Event]:
    """
        Fetch events generated from data received from the network.

        Call this method immediately after any of the ``receive_*()`` methods.

        Process resulting events, likely by passing them to the application.

        Returns:
            List[Event]: Events read from the connection.
        """
    events, self.events = (self.events, [])
    return events