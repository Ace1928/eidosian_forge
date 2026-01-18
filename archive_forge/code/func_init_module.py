import math
from dataclasses import dataclass
from typing import Optional
import torch.nn as nn
from xformers.components import Activation, build_activation
from xformers.components.feedforward import Feedforward, FeedforwardConfig
from . import register_feedforward
def init_module(m: nn.Module):
    if isinstance(m, nn.Conv2d):
        fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        fan_out //= m.groups
        m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if m.bias is not None:
            m.bias.data.zero_()