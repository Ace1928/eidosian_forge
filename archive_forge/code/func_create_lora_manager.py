import logging
from abc import ABC, abstractmethod, abstractproperty
from typing import Any, Dict, List, Optional, Set, Type
import torch
from vllm.lora.models import (LoRAModel, LoRAModelManager,
from vllm.lora.request import LoRARequest
from vllm.lora.layers import LoRAMapping
from vllm.config import LoRAConfig
def create_lora_manager(self, model: torch.nn.Module) -> Any:
    lora_manager = create_lora_manager(model, lora_manager_cls=self._lora_manager_cls, max_num_seqs=self.max_num_seqs, vocab_size=self.vocab_size, lora_config=self.lora_config, max_num_batched_tokens=self.max_num_batched_tokens)
    self._lora_manager: LRUCacheLoRAModelManager = lora_manager
    return lora_manager.model