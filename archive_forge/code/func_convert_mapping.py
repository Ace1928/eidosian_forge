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
def convert_mapping(mapping: LoRAMapping, lora_index_to_id: List[Optional[int]], max_loras: int, vocab_size: int, extra_vocab_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """Converts LoRAMapping to index tensors.

    Args:
        mapping: LoRAMapping mapping rows in a batch to LoRA ids.
        lora_index_to_id: List mapping LoRA ids to LoRA indices.
        max_loras: Maximum number of LoRAs.
        vocab_size: Model vocab size.
        extra_vocab_size: Extra vocab size each LoRA can have.

    Returns:
        A tuple of tensors:
            base_indices: Tensor of shape [batch_size] mapping batch rows to
                LoRA indices.
            sampler_indices: Tensor of shape [batch_size] mapping requests to
                LoRA indices for sampler. For generation, this will be the
                same as base_indicies. For prefill, this will map requests
                to LoRA indices.
            sampler_indices_padded: Tensor of shape [batch_size] mapping
                requests to LoRA indices for sampler with padding.
                Same as sampler_indicies, but -1 is replaced with
                max_loras.
            embeddings_indices: Tensor of shape [2, batch_size] mapping
                requests to embedding indices. First row is for embeddings
                added by the LoRAs, second row is for the LoRA.lora_a
                embeddings.
            indices_len: List of lengths of the above tensors.
    """
    indices = list(mapping.index_mapping).copy()
    embedding_indices = indices.copy()
    lora_indices = indices.copy()
    prompt_mapping = [lora_index_to_id.index(x) if x > 0 else -1 for x in mapping.prompt_mapping]
    lora_idx = None
    for i in range(len(indices)):
        lora_idx = lora_index_to_id.index(indices[i]) if indices[i] > 0 else -1
        embedding_indices[i] = lora_idx if indices[i] > 0 else 0
        indices[i] = i
        lora_indices[i] = lora_idx
    indices = torch.tensor([indices, lora_indices, embedding_indices], dtype=torch.long, device='cuda')
    prompt_mapping = torch.tensor(prompt_mapping, device='cuda', dtype=torch.long)
    embeddings_indices = torch.stack([indices[2] * extra_vocab_size, indices[2] * (vocab_size + extra_vocab_size)])
    embeddings_indices[embeddings_indices == -1] = max_loras - 1
    base_indices = indices[1]
    sampler_indices = prompt_mapping
    sampler_indices_padded = sampler_indices.clone()
    sampler_indices_padded[sampler_indices_padded == -1] = max_loras - 1
    sampler_indices_padded = torch.arange(0, len(sampler_indices_padded), device='cuda', dtype=torch.long) + sampler_indices_padded * len(sampler_indices_padded)
    indices_len = (base_indices.shape[-1], sampler_indices.shape[-1], sampler_indices_padded.shape[-1], embeddings_indices.shape[-1])
    return (base_indices, sampler_indices, sampler_indices_padded, embeddings_indices, indices_len)