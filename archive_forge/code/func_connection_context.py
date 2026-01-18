import contextvars
import importlib
import itertools
import json
import logging
import pathlib
import typing
from collections import defaultdict
from contextlib import asynccontextmanager
from contextlib import contextmanager
from dataclasses import dataclass
import trio
from trio_websocket import ConnectionClosed as WsConnectionClosed
from trio_websocket import connect_websocket_url
@contextmanager
def connection_context(connection):
    """This context manager installs ``connection`` as the session context for
    the current Trio task."""
    token = _connection_context.set(connection)
    try:
        yield
    finally:
        _connection_context.reset(token)