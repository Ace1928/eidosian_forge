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
def set_default_executor(self, executor: typing.Optional[concurrent.futures.ThreadPoolExecutor]) -> None:
    if not isinstance(executor, concurrent.futures.ThreadPoolExecutor):
        raise TypeError('The executor must be a ThreadPoolExecutor')
    self._default_executor = executor