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
def set_exception_handler(self, handler: typing.Optional[typing.Callable]) -> None:
    if handler is not None and (not callable(handler)):
        raise TypeError('The handler must be a callable or None')
    self._exception_handler = handler