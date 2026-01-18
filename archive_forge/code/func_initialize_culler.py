from __future__ import annotations
import asyncio
import os
import pathlib
import typing as t
import warnings
from collections import defaultdict
from datetime import datetime, timedelta
from functools import partial, wraps
from jupyter_client.ioloop.manager import AsyncIOLoopKernelManager
from jupyter_client.multikernelmanager import AsyncMultiKernelManager, MultiKernelManager
from jupyter_client.session import Session
from jupyter_core.paths import exists
from jupyter_core.utils import ensure_async
from jupyter_events import EventLogger
from jupyter_events.schema_registry import SchemaRegistryException
from overrides import overrides
from tornado import web
from tornado.concurrent import Future
from tornado.ioloop import IOLoop, PeriodicCallback
from traitlets import (
from jupyter_server import DEFAULT_EVENTS_SCHEMA_PATH
from jupyter_server._tz import isoformat, utcnow
from jupyter_server.prometheus.metrics import KERNEL_CURRENTLY_RUNNING_TOTAL
from jupyter_server.utils import ApiPath, import_item, to_os_path
def initialize_culler(self):
    """Start idle culler if 'cull_idle_timeout' is greater than zero.

        Regardless of that value, set flag that we've been here.
        """
    if not self._initialized_culler and self.cull_idle_timeout > 0 and (self._culler_callback is None):
        _ = IOLoop.current()
        if self.cull_interval <= 0:
            self.log.warning("Invalid value for 'cull_interval' detected (%s) - using default value (%s).", self.cull_interval, self.cull_interval_default)
            self.cull_interval = self.cull_interval_default
        self._culler_callback = PeriodicCallback(self.cull_kernels, 1000 * self.cull_interval)
        self.log.info('Culling kernels with idle durations > %s seconds at %s second intervals ...', self.cull_idle_timeout, self.cull_interval)
        if self.cull_busy:
            self.log.info('Culling kernels even if busy')
        if self.cull_connected:
            self.log.info('Culling kernels even with connected clients')
        self._culler_callback.start()
    self._initialized_culler = True