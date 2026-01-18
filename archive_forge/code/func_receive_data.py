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
def receive_data(self, data: bytes) -> None:
    """
        Receive data from the network.

        After calling this method:

        - You must call :meth:`data_to_send` and send this data to the network.
        - You should call :meth:`events_received` and process resulting events.

        Raises:
            EOFError: if :meth:`receive_eof` was called earlier.

        """
    self.reader.feed_data(data)
    next(self.parser)