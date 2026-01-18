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
def reset_execution_trackers(self) -> None:
    """Resets any per-execution trackers."""
    self.task_poll_for_reply: t.Optional[asyncio.Future] = None
    self.code_cells_executed = 0
    self._display_id_map = {}
    self.widget_state: t.Dict[str, t.Dict] = {}
    self.widget_buffers: t.Dict[str, t.Dict[t.Tuple[str, ...], t.Dict[str, str]]] = {}
    self.output_hook_stack: t.Any = collections.defaultdict(list)
    self.comm_objects: t.Dict[str, t.Any] = {}