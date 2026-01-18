import itertools
from warnings import warn
import torch
import torch.cuda
from torch.autograd import (
from torch.autograd.profiler_util import (
@property
def self_cpu_time_total(self):
    """Return CPU time as the sum of self times across all events."""
    self._check_finish()
    assert self.function_events is not None
    return self.function_events.self_cpu_time_total