from __future__ import annotations
import time
import uuid
import typing
import random
import inspect
import functools
import datetime
import itertools
import asyncio
import contextlib
import async_lru
import signal
from pathlib import Path
from frozendict import frozendict
from typing import Dict, Callable, List, Any, Union, Coroutine, TypeVar, Optional, TYPE_CHECKING
from lazyops.utils.logs import default_logger
from lazyops.utils.serialization import (
from lazyops.utils.lazy import (
def split_into_n_batches(iterable: typing.Iterable[T], size: int) -> typing.Iterable[typing.List[T]]:
    """
    Splits the items into n amount of equal items

    >>> list(split_into_batches_of_size(range(11), 3))
    [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10]]
    """
    return split_into_batches(iterable, size)