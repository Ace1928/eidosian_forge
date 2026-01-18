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
def freeze_args_and_kwargs(*args, **kwargs):
    args = tuple((recursive_freeze(arg) if isinstance(arg, dict) else arg for arg in args))
    kwargs = {k: recursive_freeze(v) if isinstance(v, dict) else v for k, v in kwargs.items()}
    return (args, kwargs)