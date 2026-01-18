import copy
import json
import logging
import math
import os
import re
from typing import (Any, Callable, Dict, Hashable, List, Optional, Tuple, Type)
import safetensors.torch
import torch
from torch import nn
from vllm.config import LoRAConfig
from vllm.utils import LRUCache, in_wsl
from vllm.lora.layers import BaseLayerWithLoRA, LoRAMapping, from_layer, from_layer_sampler
from vllm.lora.lora import LoRALayerWeights, PackedLoRALayerWeights
from vllm.lora.utils import parse_fine_tuned_lora_name, replace_submodule
@classmethod
def from_local_checkpoint(cls, lora_dir: str, lora_model_id: Optional[int]=None, device: str='cuda', dtype: Optional[torch.dtype]=None, target_embedding_padding: Optional[int]=None, embedding_modules: Optional[Dict[str, str]]=None, embedding_padding_modules: Optional[List[str]]=None) -> 'LoRAModel':
    """Create a LoRAModel from a local checkpoint."""
    lora_config_path = os.path.join(lora_dir, 'adapter_config.json')
    lora_tensor_path = os.path.join(lora_dir, 'adapter_model.safetensors')
    lora_bin_file_path = os.path.join(lora_dir, 'adapter_model.bin')
    new_embeddings_tensor_path = os.path.join(lora_dir, 'new_embeddings.safetensors')
    new_embeddings_bin_file_path = os.path.join(lora_dir, 'new_embeddings.bin')
    if os.path.isfile(lora_tensor_path):
        tensors = safetensors.torch.load_file(lora_tensor_path)
    elif os.path.isfile(lora_bin_file_path):
        tensors = torch.load(lora_bin_file_path)
    else:
        raise ValueError(f"{lora_dir} doesn't contain tensors")
    embeddings = None
    if os.path.isfile(new_embeddings_tensor_path):
        embeddings = safetensors.torch.load_file(new_embeddings_tensor_path)
    elif os.path.isfile(new_embeddings_bin_file_path):
        embeddings = torch.load(new_embeddings_bin_file_path)
    with open(lora_config_path) as f:
        config = json.load(f)
    rank = config['r']
    lora_alpha = config['lora_alpha']
    return cls.from_lora_tensors(lora_model_id=get_lora_id() if lora_model_id is None else lora_model_id, rank=rank, lora_alpha=lora_alpha, tensors=tensors, device=device, dtype=dtype, embeddings=embeddings, target_embedding_padding=target_embedding_padding, embedding_modules=embedding_modules, embedding_padding_modules=embedding_padding_modules)