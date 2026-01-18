import logging
import os
import queue
import socket
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple
import torch.cuda.memory
import torch.cuda.nvtx
import torch.nn as nn
import torch.profiler
import torch.utils.hooks
def check_schedule(self, schedule: Sequence[Tuple[Any, int, int]]) -> None:
    if len(schedule) == 0:
        logger.warning('You specified empty schedule for profiling. No data will be captured.')
    pq: Any = queue.PriorityQueue()
    for cls, begin, end in schedule:
        assert begin >= 0, f'Begin step of profiler must be non-negative, found: {begin}'
        assert end > 0, f'End step of profiler must be positive, found: {end}'
        assert begin < end, f'Start must be before the end, found: begin={begin} and end={end}'
        pq.put((begin, end))
    prev_end = -1
    for begin, end in pq.queue:
        assert begin >= prev_end, 'There is some overlapping in profiler scheduling. Please do not' + ' overlap profilers by step as they may affect each other. Schedule:' + f' {schedule}'
        prev_end = end