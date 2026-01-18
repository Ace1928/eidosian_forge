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
def get_connection_context(fn_name):
    """Look up the current connection.

    If there is no current connection, raise a ``RuntimeError`` with a
    helpful message.
    """
    try:
        return _connection_context.get()
    except LookupError:
        raise RuntimeError(f'{fn_name}() must be called in a connection context.')