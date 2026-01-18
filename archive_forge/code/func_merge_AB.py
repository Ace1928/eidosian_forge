import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List
def merge_AB(self):

    def T(w):
        return w.transpose(0, 1) if self.fan_in_fan_out else w
    delta_w = F.conv1d(self.lora_A.unsqueeze(0), self.lora_B.unsqueeze(-1), groups=sum(self.enable_lora)).squeeze(0)
    return T(self.zero_pad(delta_w))