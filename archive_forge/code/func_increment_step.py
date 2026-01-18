from collections import defaultdict
from typing import Any, Dict, List, Optional
from warnings import warn
import torch
import torch.cuda
from torch._C import _get_privateuse1_backend_name
from torch._C._profiler import _ExperimentalConfig
from torch.autograd import (
from torch.autograd.profiler_util import (
from torch.futures import Future
@classmethod
def increment_step(cls, requester: str) -> int:
    """Increments the step count for the requester.

        Additionally if the max over all step counts has incremented then
        trigger the _kineto_step() returns global step count
        """
    if requester not in cls._step_dict:
        cls.init_step_count(requester)
    cls._step_dict[requester] += 1
    new_step = max(cls._step_dict.values())
    if new_step > cls._current_step:
        delta = new_step - cls._current_step
        if delta > 1:
            warn(f'Profiler step count has increased more than 1 - current_step = {cls._current_step} step dict =  {cls._step_dict}')
        for _ in range(0, delta):
            _kineto_step()
        cls._current_step = new_step
    return cls._current_step