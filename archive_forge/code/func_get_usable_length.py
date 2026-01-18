from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
from .configuration_utils import PretrainedConfig
def get_usable_length(self, new_sequence_length=None, layer_idx: Optional[int]=0) -> int:
    return self.seen_tokens