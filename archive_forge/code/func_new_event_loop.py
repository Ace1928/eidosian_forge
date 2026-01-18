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
def new_event_loop(self) -> asyncio.AbstractEventLoop:
    return QAsyncioEventLoop(self._application, quit_qapp=self._quit_qapp)