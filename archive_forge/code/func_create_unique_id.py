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
def create_unique_id(method: typing.Optional[str]='uuid4', alph_only: typing.Optional[bool]=False, length: typing.Optional[int]=None):
    """
    Creates a unique id
    args:
        method: uuid4, uuid1, uuid5, timestamp, secret
        alph_only: if True, returns a string of only alphabets
        length: if specified, returns a string of the specified length
    """
    meth = getattr(uuid, method, None)
    if not meth:
        raise ValueError(f'Invalid UUID method: {method}')
    val = str(meth())
    if alph_only:
        val = ''.join([c for c in val if c.isalpha()])
    if length:
        while len(val) < length:
            val += str(meth())
            if alph_only:
                val = ''.join([c for c in val if c.isalpha()])
            if val.endswith('-'):
                val = val[:-1]
        val = val[:length]
    return val