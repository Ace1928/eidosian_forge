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
def call_later(self, delay: typing.Union[int, float], callback: typing.Callable, *args: typing.Any, context: typing.Optional[contextvars.Context]=None) -> asyncio.TimerHandle:
    return self._call_later_impl(delay, callback, *args, context=context, is_threadsafe=False)