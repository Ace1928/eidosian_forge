import asyncio
import collections
import warnings
from typing import (
from .base_protocol import BaseProtocol
from .helpers import BaseTimerContext, TimerNoop, set_exception, set_result
from .log import internal_logger
def get_read_buffer_limits(self) -> Tuple[int, int]:
    return (self._low_water, self._high_water)