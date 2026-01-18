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
def create_task(self, coro: typing.Union[collections.abc.Generator, collections.abc.Coroutine], *, name: typing.Optional[str]=None, context: typing.Optional[contextvars.Context]=None) -> tasks.QAsyncioTask:
    if self.is_closed():
        raise RuntimeError('Event loop is closed')
    if self._task_factory is None:
        task = tasks.QAsyncioTask(coro, loop=self, name=name, context=context)
    else:
        task = self._task_factory(self, coro, context=context)
        task.set_name(name)
    return task