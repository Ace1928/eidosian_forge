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
def resume_writing(self) -> None:
    assert self._paused
    self._paused = False
    waiter = self._drain_waiter
    if waiter is not None:
        self._drain_waiter = None
        if not waiter.done():
            waiter.set_result(None)