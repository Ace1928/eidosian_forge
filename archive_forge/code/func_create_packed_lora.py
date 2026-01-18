import os
from typing import List
import pytest
import torch
from safetensors.torch import load_file
from torch import nn
from vllm.config import LoRAConfig
from vllm.lora.layers import (ColumnParallelLinearWithLoRA,
from vllm.lora.lora import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.models import (LoRAModel, LoRAModelManager,
from vllm.lora.request import LoRARequest
from vllm.lora.worker_manager import (LRUCacheWorkerLoRAManager,
from vllm.model_executor.layers.linear import RowParallelLinear
def create_packed_lora(lora_id: int, model: nn.Module, module_name, replaced_module_names, empty_replaced_module_name=None) -> LoRAModel:
    w = model.get_submodule(module_name).weight
    loras = {}
    for replaced_module_name in replaced_module_names:
        if replaced_module_name == empty_replaced_module_name:
            continue
        loras[replaced_module_name] = LoRALayerWeights(replaced_module_name, 8, 16, torch.rand([w.shape[1], 8], device='cuda'), torch.rand([8, w.shape[0] // len(replaced_module_names)], device='cuda'))
    return LoRAModel(lora_id, 8, loras)