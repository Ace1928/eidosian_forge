from __future__ import annotations
import asyncio
import json
import time
import typing as t
import weakref
from concurrent.futures import Future
from textwrap import dedent
from jupyter_client import protocol_version as client_protocol_version  # type:ignore[attr-defined]
from tornado import gen, web
from tornado.ioloop import IOLoop
from tornado.websocket import WebSocketClosedError
from traitlets import Any, Bool, Dict, Float, Instance, Int, List, Unicode, default
from jupyter_core.utils import ensure_async
from jupyter_server.transutils import _i18n
from ..websocket import KernelWebsocketHandler
from .abc import KernelWebsocketConnectionABC
from .base import (
def on_shell_reply(msg):
    """Handle nudge shell replies."""
    self.log.debug('Nudge: shell info reply received: %s', self.kernel_id)
    if not info_future.done():
        self.log.debug('Nudge: resolving shell future: %s', self.kernel_id)
        info_future.set_result(None)