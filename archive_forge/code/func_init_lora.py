from typing import List, Optional
import torch
from vllm.lora.lora import LoRALayerWeights, PackedLoRALayerWeights
def init_lora(self, module_name: str, input_dim: int, output_dim: int, rank=8, noop=False, embeddings_tensor=None):
    lora = LoRALayerWeights(module_name, rank=rank, lora_alpha=1, lora_a=torch.rand([input_dim, rank], device='cuda'), lora_b=torch.rand([rank, output_dim], device='cuda'), embeddings_tensor=embeddings_tensor)
    self.set_module_lora(module_name, lora)
    return lora