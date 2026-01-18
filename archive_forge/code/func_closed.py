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
@property
def closed(self) -> bool:
    """
        :obj:`True` when the connection is closed; :obj:`False` otherwise.

        Be aware that both :attr:`open` and :attr:`closed` are :obj:`False`
        during the opening and closing sequences.

        """
    return self.state is State.CLOSED