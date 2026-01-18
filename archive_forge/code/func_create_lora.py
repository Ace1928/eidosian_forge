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
def create_lora(lora_id: int, model: nn.Module, sub_modules: List[str]) -> LoRAModel:
    loras = {}
    for name in sub_modules:
        w = model.get_submodule(name).weight
        loras[name] = LoRALayerWeights(name, 8, 16, torch.rand([w.shape[1], 8], device='cuda'), torch.rand([8, w.shape[0]], device='cuda'))
    return LoRAModel(lora_id, 8, loras)