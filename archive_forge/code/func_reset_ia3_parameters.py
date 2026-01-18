import warnings
from typing import Any, List, Optional
import torch
import torch.nn as nn
from transformers.pytorch_utils import Conv1D
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils import transpose
def reset_ia3_parameters(self, adapter_name):
    if adapter_name in self.ia3_l.keys():
        nn.init.constant_(self.ia3_l[adapter_name], 1.0)