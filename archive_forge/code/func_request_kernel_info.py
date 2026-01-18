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
def request_kernel_info(self):
    """send a request for kernel_info"""
    try:
        future = self.kernel_manager._kernel_info_future
    except AttributeError:
        self.log.debug('Requesting kernel info from %s', self.kernel_id)
        if self.kernel_info_channel is None:
            self.kernel_info_channel = self.multi_kernel_manager.connect_shell(self.kernel_id)
        assert self.kernel_info_channel is not None
        self.kernel_info_channel.on_recv(self._handle_kernel_info_reply)
        self.session.send(self.kernel_info_channel, 'kernel_info_request')
        self.kernel_manager._kernel_info_future = self._kernel_info_future
    else:
        if not future.done():
            self.log.debug('Waiting for pending kernel_info request')
        future.add_done_callback(lambda f: self._finish_kernel_info(f.result()))
    return _ensure_future(self._kernel_info_future)