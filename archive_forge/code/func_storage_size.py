import os
import sys
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import torch
from safetensors import deserialize, safe_open, serialize, serialize_file
def storage_size(tensor: torch.Tensor) -> int:
    try:
        return tensor.untyped_storage().nbytes()
    except AttributeError:
        try:
            return tensor.storage().size() * _SIZE[tensor.dtype]
        except NotImplementedError:
            return tensor.nelement() * _SIZE[tensor.dtype]