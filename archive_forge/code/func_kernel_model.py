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
def kernel_model(self, kernel_id):
    """Return a JSON-safe dict representing a kernel

        For use in representing kernels in the JSON APIs.
        """
    self._check_kernel_id(kernel_id)
    kernel = self._kernels[kernel_id]
    model = {'id': kernel_id, 'name': kernel.kernel_name, 'last_activity': isoformat(kernel.last_activity), 'execution_state': kernel.execution_state, 'connections': self._kernel_connections.get(kernel_id, 0)}
    if getattr(kernel, 'reason', None):
        model['reason'] = kernel.reason
    return model