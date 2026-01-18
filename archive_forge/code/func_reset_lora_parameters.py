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
def reset_lora_parameters(self, adapter_name, init_lora_weights):
    if init_lora_weights is False:
        return
    if adapter_name in self.lora_A.keys():
        if init_lora_weights is True:
            nn.init.kaiming_uniform_(self.lora_A[adapter_name].weight, a=math.sqrt(5))
        elif init_lora_weights.lower() == 'gaussian':
            nn.init.normal_(self.lora_A[adapter_name].weight, std=1 / self.r[adapter_name])
        else:
            raise ValueError(f'Unknown initialization init_lora_weights={init_lora_weights!r}')
        nn.init.zeros_(self.lora_B[adapter_name].weight)
    if adapter_name in self.lora_embedding_A.keys():
        nn.init.zeros_(self.lora_embedding_A[adapter_name])
        nn.init.normal_(self.lora_embedding_B[adapter_name])