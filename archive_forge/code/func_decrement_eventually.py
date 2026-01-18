from __future__ import annotations
import os
import socket
import threading
from collections import deque
from contextlib import contextmanager
from functools import partial
from itertools import count
from uuid import NAMESPACE_OID, uuid3, uuid4, uuid5
from amqp import ChannelError, RecoverableConnectionError
from .entity import Exchange, Queue
from .log import get_logger
from .serialization import registry as serializers
from .utils.uuid import uuid
def decrement_eventually(self, n=1):
    """Decrement the value, but do not update the channels QoS.

        Note:
        ----
            The MainThread will be responsible for calling :meth:`update`
            when necessary.
        """
    with self._mutex:
        if self.value:
            self.value -= n
            if self.value < 1:
                self.value = 1
    return self.value