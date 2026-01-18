from PySide6.QtCore import (QCoreApplication, QDateTime, QDeadlineTimer,
from . import futures
from . import tasks
import asyncio
import collections.abc
import concurrent.futures
import contextvars
import enum
import os
import signal
import socket
import subprocess
import typing
import warnings
def get_event_loop(self) -> asyncio.AbstractEventLoop:
    if self._event_loop is None:
        self._event_loop = QAsyncioEventLoop(self._application)
    return self._event_loop