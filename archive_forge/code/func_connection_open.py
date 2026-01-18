from __future__ import annotations
import asyncio
import codecs
import collections
import logging
import random
import ssl
import struct
import sys
import time
import uuid
import warnings
from typing import (
from ..datastructures import Headers
from ..exceptions import (
from ..extensions import Extension
from ..frames import (
from ..protocol import State
from ..typing import Data, LoggerLike, Subprotocol
from .compatibility import asyncio_timeout
from .framing import Frame
def connection_open(self) -> None:
    """
        Callback when the WebSocket opening handshake completes.

        Enter the OPEN state and start the data transfer phase.

        """
    assert self.state is State.CONNECTING
    self.state = State.OPEN
    if self.debug:
        self.logger.debug('= connection is OPEN')
    self.transfer_data_task = self.loop.create_task(self.transfer_data())
    self.keepalive_ping_task = self.loop.create_task(self.keepalive_ping())
    self.close_connection_task = self.loop.create_task(self.close_connection())