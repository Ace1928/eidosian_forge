from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
from .configuration_utils import PretrainedConfig
@classmethod
def from_legacy_cache(cls, past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]]=None) -> 'DynamicCache':
    """Converts a cache in the legacy cache format into an equivalent `DynamicCache`."""
    cache = cls()
    if past_key_values is not None:
        for layer_idx in range(len(past_key_values)):
            key_states, value_states = past_key_values[layer_idx]
            cache.update(key_states, value_states, layer_idx)
    return cache