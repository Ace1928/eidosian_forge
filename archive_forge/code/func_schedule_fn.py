import gzip
import json
import os
import tempfile
from enum import Enum
from functools import partial
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple
from warnings import warn
import torch
import torch.autograd.profiler as prof
from torch._C import _get_privateuse1_backend_name
from torch._C._profiler import (
from torch.autograd import kineto_available, ProfilerActivity
from torch.profiler._memory_profiler import MemoryProfile, MemoryProfileTimeline
def schedule_fn(step: int) -> ProfilerAction:
    assert step >= 0
    if step < skip_first:
        return ProfilerAction.NONE
    else:
        step -= skip_first
    num_steps = wait + warmup + active
    if repeat > 0 and step / num_steps >= repeat:
        return ProfilerAction.NONE
    mod_step = step % num_steps
    if mod_step < wait:
        return ProfilerAction.NONE
    elif mod_step < wait + warmup:
        return ProfilerAction.WARMUP
    else:
        return ProfilerAction.RECORD if mod_step < num_steps - 1 else ProfilerAction.RECORD_AND_SAVE