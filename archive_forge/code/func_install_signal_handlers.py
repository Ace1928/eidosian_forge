from __future__ import annotations
import asyncio
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
from typing import TYPE_CHECKING, Sequence, Union
import click
from uvicorn.config import Config
def install_signal_handlers(self) -> None:
    if threading.current_thread() is not threading.main_thread():
        return
    loop = asyncio.get_event_loop()
    try:
        for sig in HANDLED_SIGNALS:
            loop.add_signal_handler(sig, self.handle_exit, sig, None)
    except NotImplementedError:
        for sig in HANDLED_SIGNALS:
            signal.signal(sig, self.handle_exit)