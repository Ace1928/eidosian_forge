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
def split_into_batches_with_index(iterable: typing.Iterable[T], n: int, start: typing.Optional[int]=None) -> typing.Iterable[typing.Tuple[int, List[T]]]:
    """
    Splits the items into fixed-length chunks or blocks.

    >>> list(split_into_batches_of_n(range(11), 3))
    [(0, [0, 1, 2]), (1, [3, 4, 5]), (2, [6, 7, 8]), (3, [9, 10])]
    """
    return list(enumerate(get_batches_from_generator(iterable, n), start=start or 0))