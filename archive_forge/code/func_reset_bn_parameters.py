import math
import torch
import torch.nn as nn
import torch.ao.nn.intrinsic as nni
import torch.ao.nn.qat as nnqat
import torch.nn.functional as F
from torch.nn import init
from torch.nn.utils import fuse_conv_bn_weights
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.parameter import Parameter
from typing import TypeVar
def reset_bn_parameters(self):
    self.bn.reset_running_stats()
    init.uniform_(self.bn.weight)
    init.zeros_(self.bn.bias)
    if self.bias is not None:
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in)
        init.uniform_(self.bias, -bound, bound)