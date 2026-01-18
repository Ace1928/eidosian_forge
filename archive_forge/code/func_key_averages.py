import itertools
from warnings import warn
import torch
import torch.cuda
from torch.autograd import (
from torch.autograd.profiler_util import (
def key_averages(self, group_by_input_shape=False, group_by_stack_n=0):
    self._check_finish()
    assert self.function_events is not None, 'Expected profiling results'
    return self.function_events.key_averages(group_by_input_shape, group_by_stack_n)