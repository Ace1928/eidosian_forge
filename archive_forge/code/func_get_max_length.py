from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
from .configuration_utils import PretrainedConfig
def get_max_length(self) -> Optional[int]:
    """Returns the maximum sequence length of the cached states. DynamicCache does not have a maximum length."""
    return self.max_cache_len