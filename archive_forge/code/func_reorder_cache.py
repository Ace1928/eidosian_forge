from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
from .configuration_utils import PretrainedConfig
def reorder_cache(self, beam_idx: torch.LongTensor):
    """Reorders the cache for beam search, given the selected beam indices."""
    device = self.key_cache.device
    self.key_cache = self.key_cache.index_select(0, beam_idx.to(device))
    device = self.value_cache.device
    self.value_cache = self.value_cache.index_select(0, beam_idx.to(device))