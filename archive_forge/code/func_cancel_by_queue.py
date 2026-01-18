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
def cancel_by_queue(self, queue):
    """Cancel consumer by queue name."""
    qname = queue.name if isinstance(queue, Queue) else queue
    try:
        tag = self._active_tags.pop(qname)
    except KeyError:
        pass
    else:
        self.channel.basic_cancel(tag)
    finally:
        self._queues.pop(qname, None)