from contextlib import contextmanager
from queue import Queue
import sys
from threading import Thread
from types import TracebackType
from typing import TYPE_CHECKING, Callable, Dict, Generator, List, Optional, Tuple, Type, Union, cast
import torch
from .microbatch import Batch
from .stream import AbstractStream, use_device, use_stream
def join_workers(in_queues: List[InQueue], out_queues: List[OutQueue]) -> None:
    for in_queue in set(in_queues):
        in_queue.put(None)
    running = set(out_queues)
    while running:
        out_queue = running.pop()
        ok, payload = out_queue.get()
        done = (False, None)
        if (ok, payload) == done:
            continue
        running.add(out_queue)