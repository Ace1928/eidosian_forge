from __future__ import annotations
import abc
import time
import typing
from .common import BaseScreen
def queue_command(self, command: int, data: bytearray) -> None:
    self._command_queue.append((command, data))
    if self._last_command is None:
        self._send_next_command()