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
def build_dict_from_query(query: str, **kwargs) -> Dict[str, Union[int, float, str, datetime.datetime, Any]]:
    """
    Builds a dictionary from a query
    """
    if not query.startswith('{') and (not query.startswith('[')):
        import base64
        query = base64.b64decode(query).decode('utf-8')
    data = build_dict_from_str(query, **kwargs)
    for k, v in data.items():
        if 'date' in k:
            with contextlib.suppress(Exception):
                from lazyops.utils.dates import parse_datetime
                v = parse_datetime(v)
        data[k] = v
    return data