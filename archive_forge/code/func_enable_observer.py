import torch
from torch.nn.parameter import Parameter
from typing import List
@torch.jit.export
def enable_observer(self, enabled=True):
    self.toggle_observer_update(enabled)