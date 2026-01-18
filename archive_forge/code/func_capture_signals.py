from __future__ import annotations
import asyncio
import contextlib
import logging
import os
import platform
import signal
import socket
import sys
import threading
import time
from email.utils import formatdate
from types import FrameType
from typing import TYPE_CHECKING, Generator, Sequence, Union
import click
from uvicorn.config import Config
@contextlib.contextmanager
def capture_signals(self) -> Generator[None, None, None]:
    if threading.current_thread() is not threading.main_thread():
        yield
        return
    original_handlers = {sig: signal.signal(sig, self.handle_exit) for sig in HANDLED_SIGNALS}
    try:
        yield
    finally:
        for sig, handler in original_handlers.items():
            signal.signal(sig, handler)
    for captured_signal in reversed(self._captured_signals):
        signal.raise_signal(captured_signal)