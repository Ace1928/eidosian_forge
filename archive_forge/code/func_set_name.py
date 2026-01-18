from . import events
from . import futures
import asyncio
import collections.abc
import concurrent.futures
import contextvars
import typing
def set_name(self, value) -> None:
    self._name = str(value)