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
def send_eof(self) -> None:
    assert not self.eof_sent
    self.eof_sent = True
    if self.debug:
        self.logger.debug('> EOF')
    self.writes.append(SEND_EOF)