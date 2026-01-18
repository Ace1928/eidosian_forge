from __future__ import annotations
import json
import logging
import os
from typing import TYPE_CHECKING, Any
import tornado.websocket
from tornado import gen
from tornado.concurrent import run_on_executor
@run_on_executor(executor='_blocking_io_executor')
def stdin_to_ptyproc(self, text: str) -> None:
    """Handles stdin messages sent on the websocket.

        This is a blocking call that should NOT be performed inside the
        server primary event loop thread. Messages must be handled
        asynchronously to prevent blocking on the PTY buffer.
        """
    if self.terminal is not None:
        self.terminal.ptyproc.write(text)