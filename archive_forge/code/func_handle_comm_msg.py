import asyncio
import atexit
import base64
import collections
import datetime
import re
import signal
import typing as t
from contextlib import asynccontextmanager, contextmanager
from queue import Empty
from textwrap import dedent
from time import monotonic
from jupyter_client import KernelManager
from jupyter_client.client import KernelClient
from nbformat import NotebookNode
from nbformat.v4 import output_from_msg
from traitlets import Any, Bool, Callable, Dict, Enum, Integer, List, Type, Unicode, default
from traitlets.config.configurable import LoggingConfigurable
from .exceptions import (
from .output_widget import OutputWidget
from .util import ensure_async, run_hook, run_sync
def handle_comm_msg(self, outs: t.List, msg: t.Dict, cell_index: int) -> None:
    """Handle a comm message."""
    content = msg['content']
    data = content['data']
    if self.store_widget_state and 'state' in data:
        self.widget_state.setdefault(content['comm_id'], {}).update(data['state'])
        if 'buffer_paths' in data and data['buffer_paths']:
            comm_id = content['comm_id']
            if comm_id not in self.widget_buffers:
                self.widget_buffers[comm_id] = {}
            new_buffers: t.Dict[t.Tuple[str, ...], t.Dict[str, str]] = {tuple(k['path']): k for k in self._get_buffer_data(msg)}
            self.widget_buffers[comm_id].update(new_buffers)
    if msg['msg_type'] == 'comm_open':
        target = msg['content'].get('target_name')
        handler = self.comm_open_handlers.get(target)
        if handler:
            comm_id = msg['content']['comm_id']
            comm_object = handler(msg)
            if comm_object:
                self.comm_objects[comm_id] = comm_object
        else:
            self.log.warning(f'No handler found for comm target {target!r}')
    elif msg['msg_type'] == 'comm_msg':
        content = msg['content']
        comm_id = msg['content']['comm_id']
        if comm_id in self.comm_objects:
            self.comm_objects[comm_id].handle_msg(msg)