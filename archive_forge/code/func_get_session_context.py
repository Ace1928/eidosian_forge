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
def get_session_context(fn_name):
    """Look up the current session.

    If there is no current session, raise a ``RuntimeError`` with a
    helpful message.
    """
    try:
        return _session_context.get()
    except LookupError:
        raise RuntimeError(f'{fn_name}() must be called in a session context.')