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
def start_watching_activity(self, kernel_id):
    """Start watching IOPub messages on a kernel for activity.

        - update last_activity on every message
        - record execution_state from status messages
        """
    kernel = self._kernels[kernel_id]
    kernel.execution_state = 'starting'
    kernel.reason = ''
    kernel.last_activity = utcnow()
    kernel._activity_stream = kernel.connect_iopub()
    session = Session(config=kernel.session.config, key=kernel.session.key)

    def record_activity(msg_list):
        """Record an IOPub message arriving from a kernel"""
        self.last_kernel_activity = kernel.last_activity = utcnow()
        idents, fed_msg_list = session.feed_identities(msg_list)
        msg = session.deserialize(fed_msg_list, content=False)
        msg_type = msg['header']['msg_type']
        if msg_type == 'status':
            msg = session.deserialize(fed_msg_list)
            kernel.execution_state = msg['content']['execution_state']
            self.log.debug('activity on %s: %s (%s)', kernel_id, msg_type, kernel.execution_state)
        else:
            self.log.debug('activity on %s: %s', kernel_id, msg_type)
    kernel._activity_stream.on_recv(record_activity)