from contextlib import contextmanager
from queue import Queue
import sys
from threading import Thread
from types import TracebackType
from typing import TYPE_CHECKING, Callable, Dict, Generator, List, Optional, Tuple, Type, Union, cast
import torch
from .microbatch import Batch
from .stream import AbstractStream, use_device, use_stream
def normalize_device(device: torch.device) -> torch.device:
    if device.type == 'cuda' and device.index is None:
        return torch.device('cuda', index=torch.cuda.current_device())
    if device.type == 'cpu' and device.index is not None:
        return torch.device('cpu')
    return device