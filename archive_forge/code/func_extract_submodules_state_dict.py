import json
import os
from collections.abc import Mapping
from typing import Dict, List, Optional, Union
import numpy as np
import torch
from safetensors import safe_open
def extract_submodules_state_dict(state_dict: Dict[str, torch.Tensor], submodule_names: List[str]):
    """
    Extract the sub state-dict corresponding to a list of given submodules.

    Args:
        state_dict (`Dict[str, torch.Tensor]`): The state dict to extract from.
        submodule_names (`List[str]`): The list of submodule names we want to extract.
    """
    result = {}
    for module_name in submodule_names:
        result.update({key: param for key, param in state_dict.items() if key == module_name or key.startswith(module_name + '.')})
    return result