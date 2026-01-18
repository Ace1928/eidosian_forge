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
def scale_layer(self, scale: float) -> None:
    if scale == 1:
        return
    for active_adapter in self.active_adapters:
        if active_adapter not in self.lora_A.keys():
            continue
        self.scaling[active_adapter] *= scale