import abc
import asyncio
import datetime
import functools
import logging
import os
import random
import threading
import time
from typing import Any, Awaitable, Callable, Generic, Optional, Tuple, Type, TypeVar
from requests import HTTPError
import wandb
from wandb.util import CheckRetryFnType
from .mailbox import ContextCancelledError
def next_sleep_or_reraise(self, exc: Exception) -> datetime.timedelta:
    if not self._filter(exc):
        raise exc
    return self._wrapped.next_sleep_or_reraise(exc)