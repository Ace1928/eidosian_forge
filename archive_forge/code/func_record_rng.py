from typing import List
import torch
import torch.nn as nn
from torch.autograd.function import Function
from torch.utils.checkpoint import get_device_states, set_device_states
from xformers.components import RequiresWrappedInputs
def record_rng(self, *args):
    self.cpu_state = torch.get_rng_state()
    if torch.cuda._initialized:
        self.cuda_in_fwd = True
        self.gpu_devices, self.gpu_states = get_device_states(*args)