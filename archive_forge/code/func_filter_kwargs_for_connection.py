import os
import sys
import logging
import asyncio
import typing
import contextlib
import threading
import inspect
from itertools import chain
from queue import Empty, Full, Queue, LifoQueue
from urllib.parse import parse_qs, unquote, urlparse, ParseResult
from redis.connection import (
from redis.asyncio.connection import (
import aiokeydb.v2.exceptions as exceptions
from aiokeydb.v2.utils import set_ulimits
def filter_kwargs_for_connection(conn_cls: typing.Type[Connection], kwargs: typing.Dict[str, typing.Any]) -> typing.Dict[str, typing.Any]:
    """
    Filter out kwargs that aren't valid for a connection class
    """
    global _conn_valid_kwarg_keys
    if conn_cls.__name__ not in _conn_valid_kwarg_keys:
        _conn_valid_kwarg_keys[conn_cls.__name__] = _extract_cls_init_kwargs(conn_cls)
    return {k: v for k, v in kwargs.items() if k in _conn_valid_kwarg_keys[conn_cls.__name__]}