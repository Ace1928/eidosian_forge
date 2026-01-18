import math
import warnings
from typing import Any, List, Optional, Union
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.pytorch_utils import Conv1D
from peft.tuners.tuners_utils import BaseTunerLayer, check_adapters_to_merge
from peft.utils.integrations import gather_params_ctx
from peft.utils.other import transpose
from .config import LoraConfig
def set_scale(self, adapter, scale):
    if adapter not in self.scaling:
        return
    self.scaling[adapter] = scale * self.lora_alpha[adapter] / self.r[adapter]