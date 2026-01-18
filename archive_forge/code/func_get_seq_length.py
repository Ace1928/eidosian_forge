from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
from .configuration_utils import PretrainedConfig
def get_seq_length(self, layer_idx: Optional[int]=0) -> int:
    """Returns the sequence length of the cached states that were seen by the model. `layer_idx` kept for BC"""
    return self.seen_tokens