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
def loftq_init(self, adapter_name):
    from peft.utils.loftq_utils import loftq_init
    weight = self.get_base_layer().weight
    kwargs = {'num_bits': self.kwargs.get('loftq_bits', 4), 'reduced_rank': self.r[adapter_name], 'num_iter': self.kwargs.get('loftq_iter', 1)}
    qweight, lora_A, lora_B = loftq_init(weight, **kwargs)
    if adapter_name in self.lora_A.keys():
        self.lora_A[adapter_name].weight.data = lora_A
        self.lora_B[adapter_name].weight.data = lora_B
    if adapter_name in self.lora_embedding_A.keys():
        self.lora_embedding_A[adapter_name].weight.data = lora_A
        self.lora_embedding_B[adapter_name].weight.data = lora_B
    self.get_base_layer().weight.data = qweight