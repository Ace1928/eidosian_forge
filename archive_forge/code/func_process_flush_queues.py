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
def process_flush_queues(self, proc):
    """Flush all queues.

        Including the outbound buffer, so that
        all tasks that haven't been started will be discarded.

        In Celery this is called whenever the transport connection is lost
        (consumer restart), and when a process is terminated.
        """
    resq = proc.outq._reader
    on_state_change = self._result_handler.on_state_change
    fds = {resq}
    while fds and (not resq.closed) and (self._state != TERMINATE):
        readable, _, _ = _select(fds, None, fds, timeout=0.01)
        if readable:
            try:
                task = resq.recv()
            except (OSError, EOFError) as exc:
                _errno = getattr(exc, 'errno', None)
                if _errno == errno.EINTR:
                    continue
                elif _errno == errno.EAGAIN:
                    break
                elif _errno not in UNAVAIL:
                    debug('got %r while flushing process %r', exc, proc, exc_info=1)
                break
            else:
                if task is None:
                    debug('got sentinel while flushing process %r', proc)
                    break
                else:
                    on_state_change(task)
        else:
            break