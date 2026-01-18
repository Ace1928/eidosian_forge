from . import events
from . import futures
import asyncio
import collections.abc
import concurrent.futures
import contextvars
import typing
def cancelling(self) -> bool:
    raise NotImplementedError('QtTask.cancelling is not implemented')