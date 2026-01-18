from __future__ import annotations
from itertools import count
from typing import TYPE_CHECKING
from .common import maybe_declare
from .compression import compress
from .connection import is_connection, maybe_channel
from .entity import Exchange, Queue, maybe_delivery_mode
from .exceptions import ContentDisallowed
from .serialization import dumps, prepare_accept_content
from .utils.functional import ChannelPromise, maybe_list
def consuming_from(self, queue):
    """Return :const:`True` if currently consuming from queue'."""
    name = queue
    if isinstance(queue, Queue):
        name = queue.name
    return name in self._active_tags