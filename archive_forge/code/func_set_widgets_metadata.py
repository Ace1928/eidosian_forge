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
def set_widgets_metadata(self) -> None:
    """Set with widget metadata."""
    if self.widget_state:
        self.nb.metadata.widgets = {'application/vnd.jupyter.widget-state+json': {'state': {model_id: self._serialize_widget_state(state) for model_id, state in self.widget_state.items() if '_model_name' in state}, 'version_major': 2, 'version_minor': 0}}
        for key, widget in self.nb.metadata.widgets['application/vnd.jupyter.widget-state+json']['state'].items():
            buffers = self.widget_buffers.get(key)
            if buffers:
                widget['buffers'] = list(buffers.values())