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
def set_event_loop(self, loop: typing.Optional[asyncio.AbstractEventLoop]) -> None:
    self._event_loop = loop