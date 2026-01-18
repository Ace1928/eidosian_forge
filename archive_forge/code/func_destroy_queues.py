import errno
import gc
import inspect
import os
import select
import time
from collections import Counter, deque, namedtuple
from io import BytesIO
from numbers import Integral
from pickle import HIGHEST_PROTOCOL
from struct import pack, unpack, unpack_from
from time import sleep
from weakref import WeakValueDictionary, ref
from billiard import pool as _pool
from billiard.compat import isblocking, setblocking
from billiard.pool import ACK, NACK, RUN, TERMINATE, WorkersJoined
from billiard.queues import _SimpleQueue
from kombu.asynchronous import ERR, WRITE
from kombu.serialization import pickle as _pickle
from kombu.utils.eventio import SELECT_BAD_FD
from kombu.utils.functional import fxrange
from vine import promise
from celery.signals import worker_before_create_process
from celery.utils.functional import noop
from celery.utils.log import get_logger
from celery.worker import state as worker_state
def destroy_queues(self, queues, proc):
    """Destroy queues that can no longer be used.

        This way they can be replaced by new usable sockets.
        """
    assert not proc._is_alive()
    self._waiting_to_start.discard(proc)
    removed = 1
    try:
        self._queues.pop(queues)
    except KeyError:
        removed = 0
    try:
        self.on_inqueue_close(queues[0]._writer.fileno(), proc)
    except OSError:
        pass
    for queue in queues:
        if queue:
            for sock in (queue._reader, queue._writer):
                if not sock.closed:
                    self.hub_remove(sock)
                    try:
                        sock.close()
                    except OSError:
                        pass
    return removed