import math
from dataclasses import dataclass
from typing import List, Optional, Tuple, Union
import torch
import torch.utils.checkpoint
from torch import nn
from ...modeling_utils import PreTrainedModel
from ...utils import (
from .configuration_encodec import EncodecConfig
def get_num_quantizers_for_bandwidth(self, bandwidth: Optional[float]=None) -> int:
    """Return num_quantizers based on specified target bandwidth."""
    bw_per_q = math.log2(self.codebook_size) * self.frame_rate
    num_quantizers = self.num_quantizers
    if bandwidth is not None and bandwidth > 0.0:
        num_quantizers = int(max(1, math.floor(bandwidth * 1000 / bw_per_q)))
    return num_quantizers