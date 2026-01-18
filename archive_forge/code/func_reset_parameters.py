import math
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Union
import numpy as np
import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import functional as F
from ....modeling_utils import PreTrainedModel
from ....utils import (
from .configuration_trajectory_transformer import TrajectoryTransformerConfig
def reset_parameters(self):
    for i in range(self.n_models):
        nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight[i])
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias[i], -bound, bound)