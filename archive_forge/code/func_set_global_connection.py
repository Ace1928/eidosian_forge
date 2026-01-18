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
def set_global_connection(connection):
    """Install ``connection`` in the root context so that it will become the
    default connection for all tasks.

    This is generally not recommended, except it may be necessary in
    certain use cases such as running inside Jupyter notebook.
    """
    global _connection_context
    _connection_context = contextvars.ContextVar('_connection_context', default=connection)