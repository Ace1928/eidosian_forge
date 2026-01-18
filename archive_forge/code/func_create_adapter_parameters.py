import math
from typing import Any, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lycoris_utils import LycorisLayer
def create_adapter_parameters(self, adapter_name: str, r: int, shape: Tuple[int, ...]):
    if len(shape) == 4:
        self.hada_t1[adapter_name] = nn.Parameter(torch.empty(r, r, shape[2], shape[3]))
        self.hada_w1_a[adapter_name] = nn.Parameter(torch.empty(r, shape[0]))
        self.hada_w1_b[adapter_name] = nn.Parameter(torch.empty(r, shape[1]))
        self.hada_t2[adapter_name] = nn.Parameter(torch.empty(r, r, shape[2], shape[3]))
        self.hada_w2_a[adapter_name] = nn.Parameter(torch.empty(r, shape[0]))
        self.hada_w2_b[adapter_name] = nn.Parameter(torch.empty(r, shape[1]))
    else:
        self.hada_w1_a[adapter_name] = nn.Parameter(torch.empty(shape[0], r))
        self.hada_w1_b[adapter_name] = nn.Parameter(torch.empty(r, shape[1]))
        self.hada_w2_a[adapter_name] = nn.Parameter(torch.empty(shape[0], r))
        self.hada_w2_b[adapter_name] = nn.Parameter(torch.empty(r, shape[1]))