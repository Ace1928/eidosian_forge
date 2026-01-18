import asyncio
import collections
import warnings
from typing import (
from .base_protocol import BaseProtocol
from .helpers import BaseTimerContext, TimerNoop, set_exception, set_result
from .log import internal_logger
def is_eof(self) -> bool:
    return self._eof