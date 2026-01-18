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
def call_at(self, when: typing.Union[int, float], callback: typing.Callable, *args: typing.Any, context: typing.Optional[contextvars.Context]=None) -> asyncio.TimerHandle:
    return self._call_at_impl(when, callback, *args, context=context, is_threadsafe=False)