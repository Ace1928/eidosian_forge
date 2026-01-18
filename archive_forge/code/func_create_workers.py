from contextlib import contextmanager
from queue import Queue
import sys
from threading import Thread
from types import TracebackType
from typing import TYPE_CHECKING, Callable, Dict, Generator, List, Optional, Tuple, Type, Union, cast
import torch
from .microbatch import Batch
from .stream import AbstractStream, use_device, use_stream
def create_workers(devices: List[torch.device]) -> Tuple[List[InQueue], List[OutQueue]]:
    """Spawns worker threads. A worker thread is bound to a device."""
    in_queues: List[InQueue] = []
    out_queues: List[OutQueue] = []
    workers: Dict[torch.device, Tuple[InQueue, OutQueue]] = {}

    def normalize_device(device: torch.device) -> torch.device:
        if device.type == 'cuda' and device.index is None:
            return torch.device('cuda', index=torch.cuda.current_device())
        if device.type == 'cpu' and device.index is not None:
            return torch.device('cpu')
        return device
    for device in devices:
        device = normalize_device(device)
        try:
            in_queue, out_queue = workers[device]
        except KeyError:
            in_queue = Queue()
            out_queue = Queue()
            workers[device] = (in_queue, out_queue)
            t = Thread(target=worker, args=(in_queue, out_queue, device), daemon=True)
            t.start()
        in_queues.append(in_queue)
        out_queues.append(out_queue)
    return (in_queues, out_queues)