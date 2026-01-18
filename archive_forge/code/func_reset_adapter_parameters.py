import math
from typing import Any, Set, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft.tuners.lycoris_utils import LycorisLayer
def reset_adapter_parameters(self, adapter_name: str):
    if adapter_name in self.hada_w1_a.keys():
        nn.init.kaiming_uniform_(self.hada_w1_a[adapter_name], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.hada_w1_b[adapter_name], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.hada_w2_a[adapter_name], a=math.sqrt(5))
        nn.init.zeros_(self.hada_w2_b[adapter_name])
    if adapter_name in self.hada_t1.keys():
        nn.init.kaiming_uniform_(self.hada_t1[adapter_name], a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.hada_t2[adapter_name], a=math.sqrt(5))