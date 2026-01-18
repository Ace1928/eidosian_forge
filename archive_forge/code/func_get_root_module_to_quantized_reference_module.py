from typing import Dict, Any, List, Callable, Union, Tuple, Type
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backend_config import (
from ..utils import Pattern
from ..fuser_method_mappings import (
def get_root_module_to_quantized_reference_module(backend_config: BackendConfig) -> Dict[Type[torch.nn.Module], Type[torch.nn.Module]]:
    mapping: Dict[Type[torch.nn.Module], Type[torch.nn.Module]] = {}
    for config in backend_config.configs:
        if config.root_module is not None and config.reference_quantized_module is not None:
            mapping[config.root_module] = config.reference_quantized_module
    return mapping