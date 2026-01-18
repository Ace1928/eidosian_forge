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
def on_process_down(proc):
    """Called when a worker process exits."""
    if getattr(proc, 'dead', None):
        return
    process_flush_queues(proc)
    _remove_from_index(proc.outq._reader, proc, fileno_to_outq, remove_reader)
    if proc.synq:
        _remove_from_index(proc.synq._writer, proc, fileno_to_synq, remove_writer)
    inq = _remove_from_index(proc.inq._writer, proc, fileno_to_inq, remove_writer, callback=all_inqueues.discard)
    if inq:
        busy_workers.discard(inq)
    self._untrack_child_process(proc, hub)
    waiting_to_start.discard(proc)
    self._active_writes.discard(proc.inqW_fd)
    remove_writer(proc.inq._writer)
    remove_reader(proc.outq._reader)
    if proc.synqR_fd:
        remove_reader(proc.synq._reader)
    if proc.synqW_fd:
        self._active_writes.discard(proc.synqW_fd)
        remove_reader(proc.synq._writer)