from __future__ import annotations
import logging # isort:skip
import sys
import threading
from collections import defaultdict
from traceback import format_exception
from typing import (
import tornado
from tornado import gen
from ..core.types import ID
def add_periodic_callback(self, callback: Callback, period_milliseconds: int, callback_id: ID) -> None:
    """ Adds a callback to be run every period_milliseconds until it is removed.

        The passed-in ID can be used with remove_periodic_callback.

        """
    cb = _AsyncPeriodic(callback, period_milliseconds, io_loop=self._loop)
    self._assign_remover(callback, callback_id, self._periodic_callback_removers, cb.stop)
    cb.start()