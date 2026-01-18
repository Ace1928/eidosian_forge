from queue import Queue
from types import TracebackType
from typing import TYPE_CHECKING, Iterable, List, Optional, Tuple, Type, Union, cast
import torch
from torch import Tensor, nn
from torch.autograd.profiler import record_function
from .checkpoint import Checkpointing
from .copy import Copy, Wait
from .dependency import fork, join
from .microbatch import Batch
from .skip.layout import SkipLayout
from .skip.tracker import SkipTrackerThroughPotals, use_skip_tracker
from .stream import AbstractStream, current_stream, use_device
from .worker import Task, create_workers, join_workers
def depend(fork_from: Batch, join_to: Batch) -> None:
    fork_from[0], phony = fork(fork_from[0])
    join_to[0] = join(join_to[0], phony)