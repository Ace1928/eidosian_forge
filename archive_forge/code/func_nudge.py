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
def nudge(count):
    """Nudge the kernel."""
    count += 1
    if self.kernel_id not in self.multi_kernel_manager:
        self.log.debug('Nudge: cancelling on stopped kernel: %s', self.kernel_id)
        finish()
        return
    if shell_channel.closed():
        self.log.debug('Nudge: cancelling on closed zmq socket: %s', self.kernel_id)
        finish()
        return
    if control_channel.closed():
        self.log.debug('Nudge: cancelling on closed zmq socket: %s', self.kernel_id)
        finish()
        return
    if not both_done.done():
        log = self.log.warning if count % 10 == 0 else self.log.debug
        log(f'Nudge: attempt {count} on kernel {self.kernel_id}')
        self.session.send(shell_channel, 'kernel_info_request')
        self.session.send(control_channel, 'kernel_info_request')
        nonlocal nudge_handle
        nudge_handle = loop.call_later(0.5, nudge, count)