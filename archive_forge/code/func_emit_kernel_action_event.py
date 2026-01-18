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
def emit_kernel_action_event(success_msg: str='') -> t.Callable[..., t.Any]:
    """Decorate kernel action methods to
    begin emitting jupyter kernel action events.

    Parameters
    ----------
    success_msg: str
        A formattable string that's passed to the message field of
        the emitted event when the action succeeds. You can include
        the kernel_id, kernel_name, or action in the message using
        a formatted string argument,
        e.g. "{kernel_id} succeeded to {action}."

    error_msg: str
        A formattable string that's passed to the message field of
        the emitted event when the action fails. You can include
        the kernel_id, kernel_name, or action in the message using
        a formatted string argument,
        e.g. "{kernel_id} failed to {action}."
    """

    def wrap_method(method):

        @wraps(method)
        async def wrapped_method(self, *args, **kwargs):
            """"""
            action = method.__name__.replace('_kernel', '')
            try:
                out = await method(self, *args, **kwargs)
                data = {'kernel_name': self.kernel_name, 'action': action, 'status': 'success', 'msg': success_msg.format(kernel_id=self.kernel_id, kernel_name=self.kernel_name, action=action)}
                if self.kernel_id:
                    data['kernel_id'] = self.kernel_id
                self.emit(schema_id='https://events.jupyter.org/jupyter_server/kernel_actions/v1', data=data)
                return out
            except Exception as err:
                data = {'kernel_name': self.kernel_name, 'action': action, 'status': 'error', 'msg': str(err)}
                if self.kernel_id:
                    data['kernel_id'] = self.kernel_id
                if isinstance(err, web.HTTPError):
                    msg = err.log_message or ''
                    data['status_code'] = err.status_code
                    data['msg'] = msg
                self.emit(schema_id='https://events.jupyter.org/jupyter_server/kernel_actions/v1', data=data)
                raise err
        return wrapped_method
    return wrap_method