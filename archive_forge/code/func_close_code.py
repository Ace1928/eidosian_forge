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
def close_code(self) -> Optional[int]:
    """
        WebSocket close code, defined in `section 7.1.5 of RFC 6455`_.

        .. _section 7.1.5 of RFC 6455:
            https://www.rfc-editor.org/rfc/rfc6455.html#section-7.1.5

        :obj:`None` if the connection isn't closed yet.

        """
    if self.state is not State.CLOSED:
        return None
    elif self.close_rcvd is None:
        return CloseCode.ABNORMAL_CLOSURE
    else:
        return self.close_rcvd.code